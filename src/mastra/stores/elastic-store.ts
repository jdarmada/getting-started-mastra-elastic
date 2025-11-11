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

/**
 * Configuration options for ElasticVector
 */
export interface ElasticVectorConfig extends ClientOptions {
    /**
     * Explicitly specify if connecting to Elasticsearch Serverless.
     * If not provided, will be auto-detected on first use.
     * 
     * Set to true for Serverless deployments to skip auto-detection.
     */
    isServerless?: boolean;
    
    /**
     * Maximum documents to count accurately when describing indices.
     * Higher values provide accurate counts but may impact performance on large indices.
     * 
     * @default 10000
     */
    maxCountAccuracy?: number;
}

/**
 * Elasticsearch adapter for Mastra's semantic recall feature.
 * 
 * Supports both standard Elasticsearch deployments (self-managed, Elastic Cloud)
 * and Elasticsearch Serverless with automatic detection and configuration.
 * 
 * @example
 * ```typescript
 * // Auto-detect deployment type
 * const vector = new ElasticVector({
 *   node: 'https://your-cluster.es.cloud',
 *   auth: { apiKey: 'your-api-key' }
 * });
 * 
 * // Explicit serverless configuration (skips auto-detection)
 * const vector = new ElasticVector({
 *   node: 'https://your-serverless.es.cloud',
 *   auth: { apiKey: 'your-api-key' },
 *   isServerless: true
 * });
 * ```
 */
export class ElasticVector extends MastraVector {
    private client: Client;
    private isServerless: boolean | undefined;
    private deploymentChecked: boolean = false;
    private readonly maxCountAccuracy: number;

    constructor(config: ElasticVectorConfig) {
        super();
        this.client = new Client(config);
        this.isServerless = config.isServerless;
        this.maxCountAccuracy = config.maxCountAccuracy ?? 10000;
    }

    /**
     * Detects if connected to Elasticsearch Serverless.
     * 
     * Detection strategy:
     * 1. Use explicit configuration if provided
     * 2. Query cluster info for serverless indicators
     * 3. Cache result to avoid repeated API calls
     * 
     * @returns true if serverless deployment, false for standard deployments
     * @private
     */
    private async detectServerless(): Promise<boolean> {
        // Return cached result if already detected
        if (this.deploymentChecked) {
            return this.isServerless ?? false;
        }

        // Use explicit configuration if provided
        if (this.isServerless !== undefined) {
            this.deploymentChecked = true;
            this.logger?.info(
                `Using explicit deployment type: ${this.isServerless ? 'Serverless' : 'Standard'}`
            );
            return this.isServerless;
        }

        try {
            const info = await this.client.info();
            
            // Primary detection: build flavor (most reliable)
            const isBuildFlavorServerless = info.version?.build_flavor === 'serverless';
            
            // Secondary detection: tagline (fallback)
            const isTaglineServerless = info.tagline?.toLowerCase().includes('serverless') ?? false;
            
            this.isServerless = isBuildFlavorServerless || isTaglineServerless;
            this.deploymentChecked = true;
            
            this.logger?.info(
                `Auto-detected ${this.isServerless ? 'Serverless' : 'Standard'} Elasticsearch deployment`,
                { 
                    buildFlavor: info.version?.build_flavor, 
                    version: info.version?.number,
                    detectionMethod: isBuildFlavorServerless ? 'build_flavor' : 'tagline'
                }
            );
            
            return this.isServerless;
        } catch (error) {
            this.logger?.warn(
                'Could not auto-detect deployment type, assuming Standard Elasticsearch. ' +
                'Set isServerless: true explicitly in config if using Serverless.',
                { error: error instanceof Error ? error.message : String(error) }
            );
            this.isServerless = false;
            this.deploymentChecked = true;
            return false;
        }
    }

    /**
     * Creates a new vector index with the specified configuration.
     * 
     * Automatically configures index settings based on deployment type:
     * - Standard: Configures shards and replicas
     * - Serverless: Omits shard configuration (managed automatically)
     * 
     * @param params - Index configuration parameters
     * @throws Error if index creation fails or validation of existing index fails
     */
    async createIndex(params: CreateIndexParams): Promise<void> {
        const { indexName, dimension, metric = 'cosine' } = params;

        try {
            const exists = await this.client.indices.exists({ index: indexName });

            if (exists) {
                try {
                    await this.validateExistingIndex(indexName, dimension, metric);
                    this.logger?.info(`Index "${indexName}" already exists and is valid`);
                    return;
                } catch (validationError) {
                    throw new Error(
                        `Index "${indexName}" exists but does not match the required configuration: ${
                            validationError instanceof Error ? validationError.message : String(validationError)
                        }`
                    );
                }
            }

            const isServerless = await this.detectServerless();
            const similarity = this.mapMetricToSimilarity(metric);

            const indexConfig: any = {
                index: indexName,
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
                            // Dynamic mapping allows flexible metadata structures
                            // Note: In production, consider explicit field mappings for better control
                            dynamic: true,
                        },
                    },
                },
            };

            // Only configure shards/replicas for non-serverless deployments
            // Serverless manages infrastructure automatically
            if (!isServerless) {
                indexConfig.settings = {
                    number_of_shards: 1,
                    number_of_replicas: 0, // Increase for production HA deployments
                };
            }

            await this.client.indices.create(indexConfig);

            this.logger?.info(
                `Created ${isServerless ? 'Serverless' : 'Standard'} Elasticsearch index "${indexName}"`,
                { dimension, metric, similarity }
            );
        } catch (error) {
            const errorMessage = error instanceof Error ? error.message : String(error);
            this.logger?.error(`Failed to create index "${indexName}": ${errorMessage}`);
            throw new Error(`Failed to create index "${indexName}": ${errorMessage}`);
        }
    }

    /**
     * Upserts (inserts or updates) vectors into the index.
     * 
     * @param params - Vectors, metadata, and optional IDs
     * @returns Array of vector IDs (generated if not provided)
     */
    async upsert(params: UpsertVectorParams): Promise<string[]> {
        const { indexName, vectors, metadata = [], ids } = params;

        try {
            // Generate unique IDs if not provided
            const vectorIds = ids || vectors.map((_, i) => `vec_${Date.now()}_${i}_${Math.random().toString(36).substr(2, 9)}`);

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
                const erroredItems = response.items.filter((item: any) => item.index?.error);
                const erroredIds = erroredItems.map((item: any) => item.index?._id);
                const errorDetails = erroredItems.slice(0, 3).map((item: any) => ({
                    id: item.index?._id,
                    error: item.index?.error?.reason || item.index?.error,
                    type: item.index?.error?.type
                }));
                
                const errorMessage = `Failed to upsert ${erroredIds.length}/${vectors.length} vectors`;
                console.error(`${errorMessage}. Sample errors:`, JSON.stringify(errorDetails, null, 2));
                this.logger?.error(errorMessage, { 
                    failedCount: erroredIds.length, 
                    totalCount: vectors.length,
                    sampleErrors: errorDetails 
                });
                
                // Still return successfully inserted IDs
                const successfulIds = vectorIds.filter((id, idx) => 
                    !erroredIds.includes(id)
                );
                
                if (successfulIds.length === 0) {
                    throw new Error(`${errorMessage}. All operations failed. See logs for details.`);
                }
                
                return successfulIds;
            }

            this.logger?.info(`Successfully upserted ${vectors.length} vectors to "${indexName}"`);
            return vectorIds;
        } catch (error) {
            const errorMessage = error instanceof Error ? error.message : String(error);
            this.logger?.error(`Failed to upsert vectors to "${indexName}": ${errorMessage}`);
            throw new Error(`Failed to upsert vectors to "${indexName}": ${errorMessage}`);
        }
    }

    /**
     * Queries the index for similar vectors using k-NN search.
     * 
     * @param params - Query vector, number of results, optional filters
     * @returns Array of similar vectors with scores and metadata
     */
    async query(params: QueryVectorParams<any>): Promise<QueryResult[]> {
        const { indexName, queryVector, topK = 10, filter, includeVector = false } = params;

        try {
            const knnQuery: any = {
                field: 'vector',
                query_vector: queryVector,
                k: topK,
                num_candidates: Math.max(topK * 10, 100), // Search more candidates for better recall
            };

            // Apply metadata filters if provided
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

            const results = response.hits.hits.map((hit: any) => ({
                id: hit._id,
                score: hit._score || 0,
                metadata: hit._source?.metadata || {},
                vector: includeVector ? hit._source?.vector : undefined,
            }));

            this.logger?.debug(`Query returned ${results.length} results from "${indexName}"`);
            return results;
        } catch (error) {
            const errorMessage = error instanceof Error ? error.message : String(error);
            this.logger?.error(`Failed to query vectors from "${indexName}": ${errorMessage}`);
            throw new Error(`Failed to query vectors from "${indexName}": ${errorMessage}`);
        }
    }

    /**
     * Lists all non-system indices in the cluster.
     * 
     * @returns Array of index names (excludes system indices starting with '.')
     */
    async listIndexes(): Promise<string[]> {
        try {
            const response = await this.client.cat.indices({
                format: 'json',
            });

            const indices = response
                .map((index: any) => index.index)
                .filter((name: string) => !name.startsWith('.')); // Exclude system indices

            this.logger?.debug(`Found ${indices.length} user indices`);
            return indices;
        } catch (error) {
            const errorMessage = error instanceof Error ? error.message : String(error);
            this.logger?.error(`Failed to list indexes: ${errorMessage}`);
            throw new Error(`Failed to list indexes: ${errorMessage}`);
        }
    }

    /**
     * Describes an index, returning its configuration and statistics.
     * 
     * Uses deployment-appropriate APIs:
     * - Standard: Uses stats API with fallback to count
     * - Serverless: Uses count API with fallback to search
     * 
     * @param params - Index name to describe
     * @returns Index statistics including dimension, metric, and vector count
     * @throws Error if index doesn't exist or isn't configured for vector search
     */
    async describeIndex(params: DescribeIndexParams): Promise<IndexStats> {
        const { indexName } = params;

        try {
            const isServerless = await this.detectServerless();
            
            // Get mappings (works in all deployment types)
            const mappings = await this.client.indices.getMapping({ index: indexName });
            const indexMappings = mappings[indexName]?.mappings;
            const vectorField = indexMappings?.properties?.vector;

            if (!vectorField || vectorField.type !== 'dense_vector') {
                throw new Error(
                    `Index "${indexName}" is not configured for vector search. ` +
                    `Expected a 'vector' field of type 'dense_vector', ` +
                    `but ${!vectorField ? 'field not found' : `found type '${vectorField.type}'`}`
                );
            }

            if (!vectorField.dims) {
                throw new Error(
                    `Index "${indexName}" has a 'vector' field but dimensions are not specified`
                );
            }

            const dimension = vectorField.dims;
            const similarity = vectorField.similarity || 'cosine';
            const metric = this.mapSimilarityToMetric(similarity);

            // Get vector count using deployment-appropriate method
            let vectorCount = 0;
            
            if (isServerless) {
                // Serverless: Use count API (lightweight and fully supported)
                try {
                    const countResponse = await this.client.count({ index: indexName });
                    vectorCount = countResponse.count || 0;
                    this.logger?.debug(`Retrieved count for serverless index "${indexName}": ${vectorCount}`);
                } catch (countError) {
                    // Fallback: Search with size 0 for total hits
                    this.logger?.warn(
                        `Count API failed for serverless index "${indexName}", using search fallback`,
                        { error: countError instanceof Error ? countError.message : String(countError) }
                    );
                    
                    const searchResponse = await this.client.search({
                        index: indexName,
                        size: 0,
                        track_total_hits: this.maxCountAccuracy,
                    });
                    
                    vectorCount = typeof searchResponse.hits.total === 'number' 
                        ? searchResponse.hits.total 
                        : searchResponse.hits.total?.value || 0;
                    
                    // Warn if count is approximate
                    if (searchResponse.hits.total && typeof searchResponse.hits.total === 'object' && 
                        searchResponse.hits.total.relation === 'gte') {
                        this.logger?.warn(
                            `Count for "${indexName}" is approximate (≥${vectorCount}). ` +
                            `Increase maxCountAccuracy for exact counts on large indices.`
                        );
                    }
                }
            } else {
                // Standard: Try stats API first (provides detailed shard info)
                try {
                    const stats = await this.client.indices.stats({ index: indexName });
                    const indexStats = stats.indices?.[indexName];
                    vectorCount = indexStats?.total?.docs?.count || 0;
                    this.logger?.debug(`Retrieved stats for index "${indexName}": ${vectorCount} documents`);
                } catch (statsError) {
                    // Fallback: Use count API if stats unavailable
                    this.logger?.warn(
                        `Stats API failed for index "${indexName}", using count fallback`,
                        { error: statsError instanceof Error ? statsError.message : String(statsError) }
                    );
                    
                    const countResponse = await this.client.count({ index: indexName });
                    vectorCount = countResponse.count || 0;
                }
            }

            return {
                dimension,
                metric,
                count: vectorCount,
            };
        } catch (error) {
            const errorMessage = error instanceof Error ? error.message : String(error);
            this.logger?.error(`Failed to describe index "${indexName}": ${errorMessage}`);
            throw new Error(`Failed to describe index "${indexName}": ${errorMessage}`);
        }
    }

    /**
     * Deletes an index and all its data.
     * 
     * @param params - Index name to delete
     * @throws Error if deletion fails
     */
    async deleteIndex(params: DeleteIndexParams): Promise<void> {
        const { indexName } = params;

        try {
            await this.client.indices.delete({ index: indexName });
            this.logger?.info(`Successfully deleted index "${indexName}"`);
        } catch (error) {
            const errorMessage = error instanceof Error ? error.message : String(error);
            this.logger?.error(`Failed to delete index "${indexName}": ${errorMessage}`);
            throw new Error(`Failed to delete index "${indexName}": ${errorMessage}`);
        }
    }

    /**
     * Updates a specific vector by ID.
     * 
     * @param params - Index name, vector ID, and fields to update
     * @throws Error if update fails or vector doesn't exist
     */
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

            if (Object.keys(updateBody).length === 0) {
                this.logger?.warn(`Update called for vector "${id}" with no changes`);
                return;
            }

            await this.client.update({
                index: indexName,
                id,
                body: {
                    doc: updateBody,
                },
                refresh: true,
            });

            this.logger?.info(`Successfully updated vector "${id}" in index "${indexName}"`);
        } catch (error) {
            const errorMessage = error instanceof Error ? error.message : String(error);
            this.logger?.error(`Failed to update vector "${id}" in "${indexName}": ${errorMessage}`);
            throw new Error(`Failed to update vector "${id}" in "${indexName}": ${errorMessage}`);
        }
    }

    /**
     * Deletes a specific vector by ID.
     * 
     * @param params - Index name and vector ID to delete
     * @throws Error if deletion fails or vector doesn't exist
     */
    async deleteVector(params: DeleteVectorParams): Promise<void> {
        const { indexName, id } = params;

        try {
            await this.client.delete({
                index: indexName,
                id,
                refresh: true,
            });

            this.logger?.info(`Successfully deleted vector "${id}" from index "${indexName}"`);
        } catch (error) {
            const errorMessage = error instanceof Error ? error.message : String(error);
            this.logger?.error(`Failed to delete vector "${id}" from "${indexName}": ${errorMessage}`);
            throw new Error(`Failed to delete vector "${id}" from "${indexName}": ${errorMessage}`);
        }
    }

    /**
     * Maps Mastra metric names to Elasticsearch similarity functions.
     * 
     * @private
     */
    private mapMetricToSimilarity(metric: string): estypes.MappingDenseVectorSimilarity {
        const metricMap: Record<string, estypes.MappingDenseVectorSimilarity> = {
            cosine: 'cosine',
            euclidean: 'l2_norm',
            dotproduct: 'dot_product',
            dot_product: 'dot_product',
        };

        const similarity = metricMap[metric.toLowerCase()];
        
        if (!similarity) {
            this.logger?.warn(
                `Unknown metric "${metric}", defaulting to 'cosine'. ` +
                `Supported metrics: ${Object.keys(metricMap).join(', ')}`
            );
            return 'cosine';
        }

        return similarity;
    }

    /**
     * Maps Elasticsearch similarity functions to Mastra metric names.
     * 
     * @private
     */
    private mapSimilarityToMetric(similarity: string): 'cosine' | 'euclidean' | 'dotproduct' {
        const similarityMap: Record<string, 'cosine' | 'euclidean' | 'dotproduct'> = {
            cosine: 'cosine',
            l2_norm: 'euclidean',
            dot_product: 'dotproduct',
        };

        return similarityMap[similarity] || 'cosine';
    }

    /**
     * Converts Mastra filter format to Elasticsearch query DSL.
     * 
     * Supports:
     * - Simple equality: { field: value }
     * - Explicit operators: { field: { $eq: value } }
     * - In operator: { field: { $in: [value1, value2] } }
     * - Not equal: { field: { $ne: value } }
     * 
     * Note: Uses .keyword suffix for exact matching on text fields
     * 
     * @private
     */
    private buildElasticFilter(filter: any): any {
        if (!filter || typeof filter !== 'object') {
            return undefined;
        }

        const must: any[] = [];

        for (const [key, value] of Object.entries(filter)) {
            if (value === null || value === undefined) {
                continue;
            }

            if (typeof value === 'object' && !Array.isArray(value)) {
                // Handle operator-based filters
                if ('$eq' in value) {
                    must.push({ term: { [`metadata.${key}.keyword`]: value.$eq } });
                } else if ('$in' in value && Array.isArray(value.$in)) {
                    must.push({ terms: { [`metadata.${key}.keyword`]: value.$in } });
                } else if ('$ne' in value) {
                    must.push({ 
                        bool: { 
                            must_not: { 
                                term: { [`metadata.${key}.keyword`]: value.$ne } 
                            } 
                        } 
                    });
                } else if ('$gt' in value) {
                    must.push({ range: { [`metadata.${key}`]: { gt: value.$gt } } });
                } else if ('$gte' in value) {
                    must.push({ range: { [`metadata.${key}`]: { gte: value.$gte } } });
                } else if ('$lt' in value) {
                    must.push({ range: { [`metadata.${key}`]: { lt: value.$lt } } });
                } else if ('$lte' in value) {
                    must.push({ range: { [`metadata.${key}`]: { lte: value.$lte } } });
                } else {
                    this.logger?.warn(`Unsupported filter operator for field "${key}":`, value);
                }
            } else {
                // Simple equality - use .keyword for exact match on text fields
                must.push({ term: { [`metadata.${key}.keyword`]: value } });
            }
        }

        return must.length > 0 ? { bool: { must } } : undefined;
    }
}