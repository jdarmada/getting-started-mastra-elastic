// src/mastra/agents/knowledge-agent.ts
import { Agent } from '@mastra/core/agent';
import { Memory } from '@mastra/memory';
import { openai } from '@ai-sdk/openai';
import { ElasticVector } from '../stores/elastic-store';
import dotenv from "dotenv";

dotenv.config();

const ELASTICSEARCH_ENDPOINT = process.env.ELASTICSEARCH_ENDPOINT ?? "";
const ELASTICSEARCH_API_KEY = process.env.ELASTICSEARCH_API_KEY ?? "";

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
