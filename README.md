# Knowledge Agent with Elasticsearch & Mastra

An unofficial integration showcasing **Elasticsearch** as a vector database for **Mastra's** semantic recall feature. This project demonstrates how to build a knowledge assistant that remembers context across conversations using Elasticsearch's vector search capabilities.

This knowledge agent uses:
- **Mastra** - AI agent framework with memory capabilities
- **Elasticsearch** - Vector database for semantic search and recall
- **OpenAI** - GPT-4o for language understanding and text embeddings
- **Semantic Recall** - Retrieves contextually relevant past messages to enhance responses


## What you need

- **Node.js** v18+
- **Elasticsearch** instance (version 8.15 or newer)
- **Elasticsearch API Key**
- **OpenAI API key**

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/jdarmada/getting-started-mastra-elastic.git
```

### 2. Install Dependencies

```bash
npm install
```

### 3. Set Up Elasticsearch

You'll need an Elasticsearch instance. Choose one of these options:

#### Option A: Elasticsearch Cloud (Easiest)
1. Sign up at [elastic.co/cloud](https://cloud.elastic.co)
2. Create a deployment
3. Get your Cloud ID and API Key from the deployment dashboard

#### Option B: Local Elasticsearch with Docker
```bash
docker run -d \
  --name elasticsearch \
  -p 9200:9200 \
  -e "discovery.type=single-node" \
  -e "xpack.security.enabled=false" \
  docker.elastic.co/elasticsearch/elasticsearch:9.2.0
```

#### Option C: Self-Hosted
Follow the [official Elasticsearch installation guide](https://www.elastic.co/guide/en/elasticsearch/reference/current/install-elasticsearch.html)

### 4. Configure Environment Variables

In the root directory, rename the file `.env.example` to `.env` or create one if you don't see one. Replace the example values with your own credentials.

```env
ELASTICSEARCH_ENDPOINT="https://your-deployment.es.us-west-2.aws.elastic.cloud.com"

ELASTICSEARCH_API_KEY="a1B2c3D4E5f6g7H8i9J0k1L2m3N4o5P6"

OPENAI_API_KEY="sk-proj-1234abcd5678efgh90342ijkl"
```


### 5. Run the Agent

#### Start the Mastra dev server
```bash
npm run dev
```

## Usage

Once running, you'll be able to chat with your agent using Mastra's built-in agent playground called Mastra Studio. The agent:

1. **Receives your messages** and processes them with GPT-4o
2. **Stores conversation context** as vector embeddings in Elasticsearch
3. **Recalls relevant past messages** using semantic search (top 3 matches)
4. **Includes surrounding context** (2 messages before/after each match)
5. **Provides informed responses** based on conversation history

## Architecture

```
src/
│  └── mastra/
│       ├── agents/
│       │   └── knowledge-agent.ts    # Main agent configuration
│       ├── stores/
│       │   └── elastic-store.ts      # Elasticsearch vector store implementation
│       └── index.ts                  # Mastra initialization
├── package.json
├── tsconfig.json
└── .env                              # Your environment variables
```

## Customization

### Change the AI Model

Edit `src/mastra/agents/knowledge-agent.ts`:

```typescript
model: openai('gpt-4o-mini'), // Use a different model
```

### Adjust Memory Settings

Modify the semantic recall parameters:

```typescript
semanticRecall: {
    topK: 5,         // Retrieve more matches
    messageRange: 3, // Include more surrounding context
    scope: 'resource'
}
```

### Use a Different Embedding Model

Change the embedder in the Memory configuration:

```typescript
embedder: 'openai/text-embedding-3-large', // More powerful embeddings
```

## How It Works

### ElasticVector Store

The custom `ElasticVector` class (in `src/mastra/stores/elastic-store.ts`) implements Mastra's vector store interface and provides:

- **Index Management**: Creates Elasticsearch indices with dense vector fields
- **Vector Operations**: Upserts, queries, updates, and deletes vector embeddings
- **Semantic Search**: Uses cosine similarity for finding relevant messages
- **Metadata Filtering**: Supports filtering queries by metadata fields


## Troubleshooting

### Connection Issues

If you see "Failed to connect to Elasticsearch":
- Verify your `ELASTICSEARCH_ENDPOINT` is correct
- Check that Elasticsearch is running
- Check whether your credentials are being surfaced correctly
- Ensure your API key has proper permissions
- For local setups, verify the port (usually 9200)

## Learn More

- [Mastra Documentation](https://mastra.ai/docs)
- [Mastra's Semantic Recall Feature'](https://mastra.ai/docs/memory/semantic-recall)
- [Elasticsearch Vector Search Guide](https://www.elastic.co/guide/en/elasticsearch/reference/current/knn-search.html)
- [OpenAI Embeddings](https://platform.openai.com/docs/guides/embeddings)

## Contributing

This is an unofficial integration created to showcase Elasticsearch + Mastra capabilities. Contributions, issues, and feature requests are welcome!

## License

MIT License


---

