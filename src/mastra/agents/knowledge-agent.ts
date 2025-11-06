import { Agent } from '@mastra/core/agent';
import { Memory } from '@mastra/memory';
import { openai } from '@ai-sdk/openai';
import { Client } from '@elastic/elasticsearch';
import { ElasticVector } from '../stores/elastic-store';
import dotenv from "dotenv";

dotenv.config();

const ELASTICSEARCH_ENDPOINT = process.env.ELASTICSEARCH_ENDPOINT;
const ELASTICSEARCH_API_KEY = process.env.ELASTICSEARCH_API_KEY;

//Error check for undefined credentials
if (!ELASTICSEARCH_ENDPOINT || !ELASTICSEARCH_API_KEY) {
  throw new Error('Missing Elasticsearch credentials');
}

//Check to see if a connection can be established
const testClient = new Client({
  node: ELASTICSEARCH_ENDPOINT,
  auth: { 
    apiKey: ELASTICSEARCH_API_KEY 
  },
});

try {
  await testClient.ping();
  console.log('Connected to Elasticsearch successfully');
} catch (error: unknown) {
  if (error instanceof Error) {
    console.error('Failed to connect to Elasticsearch:', error.message);
  } else {
    console.error('Failed to connect to Elasticsearch:', error);
  }
  process.exit(1);
}

const vectorStore = new ElasticVector({
  node: ELASTICSEARCH_ENDPOINT,
  auth: {
    apiKey: ELASTICSEARCH_API_KEY,
  },
});

export const knowledgeAgent = new Agent({
    name: 'KnowledgeAgent',
    instructions: 'You are a helpful knowledge assistant.',
    model: openai('gpt-4o'),
    memory: new Memory({

        vector: vectorStore,

        //embedder used to create embeddings for each message
        embedder: 'openai/text-embedding-3-small',

        //set semantic recall options
        options: {
            semanticRecall: {
                topK: 3, // retrieve 3 similar messages
                messageRange: 2, // include 2 messages before/after each match
                scope: 'resource',
            },
        },
    }),
});
