import { MastraVector } from "@mastra/core/vector";
import type {
    CreateIndexParams,
    UpsertVectorParams,
    QueryVectorParams,
    IndexStats,
    QueryResult,
    UpdateVectorParams,
    DeleteVectorParams,
    DescribeIndexParams,
    DeleteIndexParams,
} from "@mastra/core/vector";
import { Client, type ClientOptions, estypes } from '@elastic/elasticsearch';

export class ElasticVector extends MastraVector {
    private client: Client;

    constructor(config: ClientOptions) {
        super();
        this.client = new Client(config);
    }

    async createIndex(params: CreateIndexParams): Promise<void> {
        const { indexName, dimension, metric = 'cosine' } = params;

        try {
            const exists = await this.client.indices.exists({ index: indexName });

            if (exists) {
                await this.validateExistingIndex(indexName, dimension, metric);
                return;
            }

            // Map metric to Elasticsearch similarity
            const similarity = this.mapMetricToSimilarity(metric);

            await this.client.indices.create({
                index: indexName,
                settings: {
                    number_of_shards: 1,
                    number_of_replicas: 0, // Set to 0 for single-node clusters
                },
                mappings: {
                    properties: {
                        vector: {
                            type: 'dense_vector',
                            dims: dimension,
                            index: true,
                            similarity: similarity,
                        },
                        metadata: {
                            type: 'object',
                            enabled: true,
                        },
                    },
                },
            });

            this.logger?.info(`Created Elasticsearch index "${indexName}" with ${dimension} dimensions and ${metric} metric`);
        } catch (error) {
            this.logger?.error(`Failed to create index "${indexName}": ${error}`);
            throw error;
        }
    }

    async upsert(params: UpsertVectorParams): Promise<string[]> {
        const { indexName, vectors, metadata = [], ids } = params;

        try {
            // Generate IDs if not provided
            const vectorIds = ids || vectors.map((_, i) => `vec_${Date.now()}_${i}`);

            const operations = vectors.flatMap((vec, index) => [
                { index: { _index: indexName, _id: vectorIds[index] } },
                {
                    vector: vec,
                    metadata: metadata[index] || {},
                },
            ]);

            const response = await this.client.bulk({
                refresh: true,
                operations,
            });

            if (response.errors) {
                const erroredItems = response.items
                    .filter((item: any) => item.index?.error);
                const erroredIds = erroredItems.map((item: any) => item.index?._id);
                const errorDetails = erroredItems.slice(0, 3).map((item: any) => ({
                    id: item.index?._id,
                    error: item.index?.error
                }));
                console.error(`Failed to upsert ${erroredIds.length} vectors. Sample errors:`, JSON.stringify(errorDetails, null, 2));
                this.logger?.error(`Failed to upsert ${erroredIds.length} vectors. Sample errors:`, JSON.stringify(errorDetails, null, 2));
            }

            return vectorIds;
        } catch (error) {
            this.logger?.error(`Failed to upsert vectors to "${indexName}": ${error}`);
            throw error;
        }
    }

    async query(params: QueryVectorParams<any>): Promise<QueryResult[]> {
        const { indexName, queryVector, topK = 10, filter, includeVector = false } = params;

        try {
            const knnQuery: any = {
                field: 'vector',
                query_vector: queryVector,
                k: topK,
                num_candidates: Math.max(topK * 10, 100),
            };

            // Add filter if provided
            if (filter) {
                knnQuery.filter = this.buildElasticFilter(filter);
            }

            const sourceFields = ['metadata'];
            if (includeVector) {
                sourceFields.push('vector');
            }

            const response = await this.client.search({
                index: indexName,
                knn: knnQuery,
                size: topK,
                _source: sourceFields,
            });

            return response.hits.hits.map((hit: any) => ({
                id: hit._id,
                score: hit._score || 0,
                metadata: hit._source.metadata,
                vector: includeVector ? hit._source.vector : undefined,
            }));
        } catch (error) {
            this.logger?.error(`Failed to query vectors from "${indexName}": ${error}`);
            throw error;
        }
    }

    async listIndexes(): Promise<string[]> {
        try {
            const response = await this.client.cat.indices({
                format: 'json',
            });

            return response.map((index: any) => index.index).filter((name: string) => !name.startsWith('.'));
        } catch (error) {
            this.logger?.error(`Failed to list indexes: ${error}`);
            throw error;
        }
    }

    async describeIndex(params: DescribeIndexParams): Promise<IndexStats> {
        const { indexName } = params;

        try {
            const [mappings, stats] = await Promise.all([
                this.client.indices.getMapping({ index: indexName }),
                this.client.indices.stats({ index: indexName }),
            ]);

            const indexMappings = mappings[indexName]?.mappings;
            const vectorField = indexMappings?.properties?.vector;

            if (!vectorField || vectorField.type !== 'dense_vector' || !vectorField.dims) {
                throw new Error(`Index "${indexName}" does not have a valid dense_vector field with dimensions`);
            }

            const dimension = vectorField.dims;
            const similarity = vectorField.similarity || 'cosine';
            const metric = this.mapSimilarityToMetric(similarity);

            const indexStats = stats.indices?.[indexName];
            const vectorCount = indexStats?.total?.docs?.count || 0;

            return {
                dimension,
                metric,
                count: vectorCount,
            };
        } catch (error) {
            this.logger?.error(`Failed to describe index "${indexName}": ${error}`);
            throw error;
        }
    }

    async deleteIndex(params: DeleteIndexParams): Promise<void> {
        const { indexName } = params;

        try {
            await this.client.indices.delete({ index: indexName });
            this.logger?.info(`Deleted index "${indexName}"`);
        } catch (error) {
            this.logger?.error(`Failed to delete index "${indexName}": ${error}`);
            throw error;
        }
    }

    async updateVector(params: UpdateVectorParams): Promise<void> {
        const { indexName, id, update } = params;

        try {
            const updateBody: any = {};

            if (update.vector) {
                updateBody.vector = update.vector;
            }

            if (update.metadata) {
                updateBody.metadata = update.metadata;
            }

            await this.client.update({
                index: indexName,
                id,
                body: {
                    doc: updateBody,
                },
                refresh: true,
            });

            this.logger?.info(`Updated vector "${id}" in index "${indexName}"`);
        } catch (error) {
            this.logger?.error(`Failed to update vector "${id}" in "${indexName}": ${error}`);
            throw error;
        }
    }

    async deleteVector(params: DeleteVectorParams): Promise<void> {
        const { indexName, id } = params;

        try {
            await this.client.delete({
                index: indexName,
                id,
                refresh: true,
            });

            this.logger?.info(`Deleted vector "${id}" from index "${indexName}"`);
        } catch (error) {
            this.logger?.error(`Failed to delete vector "${id}" from "${indexName}": ${error}`);
            throw error;
        }
    }

    private mapMetricToSimilarity(metric: string): estypes.MappingDenseVectorSimilarity {
        const metricMap: Record<string, estypes.MappingDenseVectorSimilarity> = {
            cosine: 'cosine',
            euclidean: 'l2_norm',
            dotproduct: 'dot_product',
            dot_product: 'dot_product',
        };

        return metricMap[metric] || 'cosine';
    }

    private mapSimilarityToMetric(similarity: string): 'cosine' | 'euclidean' | 'dotproduct' {
        const similarityMap: Record<string, 'cosine' | 'euclidean' | 'dotproduct'> = {
            cosine: 'cosine',
            l2_norm: 'euclidean',
            dot_product: 'dotproduct',
        };

        return similarityMap[similarity] || 'cosine';
    }

    private buildElasticFilter(filter: any): any {
        // Convert Mastra filter format to Elasticsearch filter format
        // Uses .keyword for exact matching on text fields
        if (!filter) return undefined;

        const must: any[] = [];

        for (const [key, value] of Object.entries(filter)) {
            if (typeof value === 'object' && value !== null) {
                // Handle complex filters
                if ('$eq' in value) {
                    must.push({ term: { [`metadata.${key}.keyword`]: value.$eq } });
                } else if ('$in' in value) {
                    must.push({ terms: { [`metadata.${key}.keyword`]: value.$in } });
                } else if ('$ne' in value) {
                    must.push({ bool: { must_not: { term: { [`metadata.${key}.keyword`]: value.$ne } } } });
                }
            } else {
                // Simple equality - use .keyword for exact match on text fields
                must.push({ term: { [`metadata.${key}.keyword`]: value } });
            }
        }

        return must.length > 0 ? { bool: { must } } : undefined;
    }
}