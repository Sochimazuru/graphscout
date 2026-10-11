# Question answering results (model: qwen2.5:1.5b)

### Q: Which methods are evaluated on HotpotQA?

**Answer:** The methods evaluated on HotpotQA include Stanford CoreNLP-OpenIE, spaCy, Graph Retrieval-Augmented Generation, Multi-Agent GraphRAG: A Text-to-Cypher Framework for Labeled Property Graphs, Empowering GraphRAG with Knowledge Filtering and Integration, and GraphRAG-FI.

- Grounding: cites 9 of 9 evidence facts (6 direct facts, not just 'paper mentions X'); 1 of 1 sentences have no citation
- Plan: Identify the methods used in the evaluation of HotpotQA, including the specific tasks, datasets, models, and concepts involved.
- Mentions found in question: HotpotQA, methods, HotpotQA evaluation, HotpotQA tasks, HotpotQA datasets
- Start nodes in graph: HotpotQA
- Routing (routed by relation words in the question: evaluatedOn): Stanford CoreNLP-OpenIE --evaluatedOn--> HotpotQA (paper wording: "evaluated on"); spaCy --evaluatedOn--> HotpotQA (paper wording: "evaluated on"); Graph Retrieval-Augmented Generation --evaluatedOn--> HotpotQA (paper wording: "evaluated on")
- Hop 1: from HotpotQA (4 edges seen), followed: Multi-Agent GraphRAG: A Text-to-Cypher Framework for Labeled Property Graphs --mentions--> HotpotQA; Empowering GraphRAG with Knowledge Filtering and Integration --mentions--> HotpotQA
- Hop 2: from Multi-Agent GraphRAG: A Text-to-Cypher Framework for Labeled Property Graphs, Empowering GraphRAG with Knowledge Filtering and Integration (6 edges seen), followed: Empowering GraphRAG with Knowledge Filtering and Integration --proposes--> GraphRAG-FI (paper wording: "proposes"); Multi-Agent GraphRAG: A Text-to-Cypher Framework for Labeled Property Graphs --mentions--> Graph Retrieval-Augmented Generation
- Hop 3: from GraphRAG-FI, Graph Retrieval-Augmented Generation (29 edges seen), followed: GraphRAG-FI --partOf--> GraphRAG-Filtering', 'GraphRAG-Integration (paper wording: "consists of"); Graph Retrieval-Augmented Generation --evaluatedOn--> Phi-4-mini (paper wording: "evaluated on")

| Evidence | Fact | Source papers |
|---|---|---|
| E1 | Stanford CoreNLP-OpenIE --evaluatedOn--> HotpotQA (paper wording: "evaluated on") | [2509.09272](https://arxiv.org/abs/2509.09272) |
| E2 | spaCy --evaluatedOn--> HotpotQA (paper wording: "evaluated on") | [2509.09272](https://arxiv.org/abs/2509.09272) |
| E3 | Graph Retrieval-Augmented Generation --evaluatedOn--> HotpotQA (paper wording: "evaluated on") | [2509.09272](https://arxiv.org/abs/2509.09272) |
| E4 | Multi-Agent GraphRAG: A Text-to-Cypher Framework for Labeled Property Graphs --mentions--> HotpotQA | [2511.08274](https://arxiv.org/abs/2511.08274) |
| E5 | Empowering GraphRAG with Knowledge Filtering and Integration --mentions--> HotpotQA | [2503.13804](https://arxiv.org/abs/2503.13804) |
| E6 | Empowering GraphRAG with Knowledge Filtering and Integration --proposes--> GraphRAG-FI (paper wording: "proposes") | [2503.13804](https://arxiv.org/abs/2503.13804) |
| E7 | Multi-Agent GraphRAG: A Text-to-Cypher Framework for Labeled Property Graphs --mentions--> Graph Retrieval-Augmented Generation | [2511.08274](https://arxiv.org/abs/2511.08274) |
| E8 | GraphRAG-FI --partOf--> GraphRAG-Filtering', 'GraphRAG-Integration (paper wording: "consists of") | [2503.13804](https://arxiv.org/abs/2503.13804) |
| E9 | Graph Retrieval-Augmented Generation --evaluatedOn--> Phi-4-mini (paper wording: "evaluated on") | [2605.20815](https://arxiv.org/abs/2605.20815) |

### Q: What does Inference-Scaled GraphRAG use, and what does it improve?

**Answer:** Inference-Scaled GraphRAG uses parallel scaling and majority voting over sampled trajectories, and sequential scaling and deep chain-of-thought graph traversal to improve multi-hop question answering performance.

- Grounding: cites 8 of 8 evidence facts (6 direct facts, not just 'paper mentions X'); 1 of 1 sentences have no citation
- Plan: Inference-Scaled GraphRAG is a method that scales the GraphRAG model to handle larger graphs by incorporating inference techniques.
- Mentions found in question: Inference-Scaled GraphRAG, Inference-Scaled GraphRAG uses, Inference-Scaled GraphRAG improves
- Start nodes in graph: Inference-Scaled GraphRAG
- Routing (routed by relation words in the question: improves, uses): Inference-Scaled GraphRAG --improves--> multi-hop question answering performance (paper wording: "improves"); Inference-Scaled GraphRAG --improves--> substantial gains over both traditional GraphRAG and prior graph traversal baselines (paper wording: "achieves"); Inference-Scaled GraphRAG --uses--> parallel scaling and majority voting over sampled trajectories (paper wording: "combines"); Inference-Scaled GraphRAG --uses--> sequential scaling and deep chain-of-thought graph traversal (paper wording: "combines")
- Hop 1: from Inference-Scaled GraphRAG (3 edges seen), followed: Inference-Scaled GraphRAG --reports--> inference-time scaling is a practical and architecture-agnostic solution for structured knowledge reasoning with LLMs (paper wording: "suggests")
- Hop 2: from inference-time scaling is a practical and architecture-agnostic solution for structured knowledge reasoning with LLMs (1 edges seen), followed: Inference Scaled GraphRAG: Improving Multi Hop Question Answering on Knowledge Graphs --mentions--> inference-time scaling is a practical and architecture-agnostic solution for structured knowledge reasoning with LLMs
- Hop 3: from Inference Scaled GraphRAG: Improving Multi Hop Question Answering on Knowledge Graphs (6 edges seen), followed: Inference Scaled GraphRAG: Improving Multi Hop Question Answering on Knowledge Graphs --proposes--> Inference-Scaled GraphRAG (paper wording: "proposes"); Inference Scaled GraphRAG: Improving Multi Hop Question Answering on Knowledge Graphs --mentions--> Graph Retrieval-Augmented Generation

| Evidence | Fact | Source papers |
|---|---|---|
| E1 | Inference-Scaled GraphRAG --improves--> multi-hop question answering performance (paper wording: "improves") | [2506.19967](https://arxiv.org/abs/2506.19967) |
| E2 | Inference-Scaled GraphRAG --improves--> substantial gains over both traditional GraphRAG and prior graph traversal baselines (paper wording: "achieves") | [2506.19967](https://arxiv.org/abs/2506.19967) |
| E3 | Inference-Scaled GraphRAG --uses--> parallel scaling and majority voting over sampled trajectories (paper wording: "combines") | [2506.19967](https://arxiv.org/abs/2506.19967) |
| E4 | Inference-Scaled GraphRAG --uses--> sequential scaling and deep chain-of-thought graph traversal (paper wording: "combines") | [2506.19967](https://arxiv.org/abs/2506.19967) |
| E5 | Inference-Scaled GraphRAG --reports--> inference-time scaling is a practical and architecture-agnostic solution for structured knowledge reasoning with LLMs (paper wording: "suggests") | [2506.19967](https://arxiv.org/abs/2506.19967) |
| E6 | Inference Scaled GraphRAG: Improving Multi Hop Question Answering on Knowledge Graphs --mentions--> inference-time scaling is a practical and architecture-agnostic solution for structured knowledge reasoning with LLMs | [2506.19967](https://arxiv.org/abs/2506.19967) |
| E7 | Inference Scaled GraphRAG: Improving Multi Hop Question Answering on Knowledge Graphs --proposes--> Inference-Scaled GraphRAG (paper wording: "proposes") | [2506.19967](https://arxiv.org/abs/2506.19967) |
| E8 | Inference Scaled GraphRAG: Improving Multi Hop Question Answering on Knowledge Graphs --mentions--> Graph Retrieval-Augmented Generation | [2506.19967](https://arxiv.org/abs/2506.19967) |

### Q: What is GraphRAG compared with, and which approach outperforms it?

**Answer:** GraphRAG is compared with other methods such as Vector RAG, BEACON-SP, TagRAG, and Stanford CoreNLP-OpenIE. It outperforms these methods on tasks like HotpotQA and Phi-4-mini.

- Grounding: cites 9 of 9 evidence facts (7 direct facts, not just 'paper mentions X'); 2 of 2 sentences have no citation
- Plan: Research Papers: GraphRAG is a method for analyzing research papers. Graph Neural Networks (GNNs) are a type of neural network used for graph data. The question asks about the performance of GraphRAG compared to GNNs.
- Mentions found in question: GraphRAG, Graph Neural Networks (GNNs), Research Papers
- Start nodes in graph: Graph Retrieval-Augmented Generation
- Routing (routed by relation words in the question: comparedWith, outperforms): Vector RAG --outperforms--> Graph Retrieval-Augmented Generation (paper wording: "outperforms"); BEACON-SP --comparedWith--> Graph Retrieval-Augmented Generation (paper wording: "compared with"); TagRAG --comparedWith--> Graph Retrieval-Augmented Generation (paper wording: "compares")
- Hop 1: from Graph Retrieval-Augmented Generation (26 edges seen), followed: Graph Retrieval-Augmented Generation --evaluatedOn--> HotpotQA (paper wording: "evaluated on"); Graph Retrieval-Augmented Generation --evaluatedOn--> Phi-4-mini (paper wording: "evaluated on")
- Hop 2: from HotpotQA, Phi-4-mini (7 edges seen), followed: Stanford CoreNLP-OpenIE --evaluatedOn--> HotpotQA (paper wording: "evaluated on"); Towards Robust Retrieval-Augmented Generation Based on Knowledge Graph: A Comparative Analysis --mentions--> HotpotQA
- Hop 3: from Stanford CoreNLP-OpenIE, Towards Robust Retrieval-Augmented Generation Based on Knowledge Graph: A Comparative Analysis (7 edges seen), followed: Stanford CoreNLP-OpenIE --evaluatedOn--> knowledge graph-based question answering (paper wording: "evaluated on"); Towards Robust Retrieval-Augmented Generation Based on Knowledge Graph: A Comparative Analysis --mentions--> Factual hallucinations

| Evidence | Fact | Source papers |
|---|---|---|
| E1 | Vector RAG --outperforms--> Graph Retrieval-Augmented Generation (paper wording: "outperforms") | [2507.03608](https://arxiv.org/abs/2507.03608) |
| E2 | BEACON-SP --comparedWith--> Graph Retrieval-Augmented Generation (paper wording: "compared with") | [2610.09026](https://arxiv.org/abs/2610.09026) |
| E3 | TagRAG --comparedWith--> Graph Retrieval-Augmented Generation (paper wording: "compares") | [2601.05254](https://arxiv.org/abs/2601.05254) |
| E4 | Graph Retrieval-Augmented Generation --evaluatedOn--> HotpotQA (paper wording: "evaluated on") | [2509.09272](https://arxiv.org/abs/2509.09272) |
| E5 | Graph Retrieval-Augmented Generation --evaluatedOn--> Phi-4-mini (paper wording: "evaluated on") | [2605.20815](https://arxiv.org/abs/2605.20815) |
| E6 | Stanford CoreNLP-OpenIE --evaluatedOn--> HotpotQA (paper wording: "evaluated on") | [2509.09272](https://arxiv.org/abs/2509.09272) |
| E7 | Towards Robust Retrieval-Augmented Generation Based on Knowledge Graph: A Comparative Analysis --mentions--> HotpotQA | [2603.05698](https://arxiv.org/abs/2603.05698) |
| E8 | Stanford CoreNLP-OpenIE --evaluatedOn--> knowledge graph-based question answering (paper wording: "evaluated on") | [2509.09272](https://arxiv.org/abs/2509.09272) |
| E9 | Towards Robust Retrieval-Augmented Generation Based on Knowledge Graph: A Comparative Analysis --mentions--> Factual hallucinations | [2603.05698](https://arxiv.org/abs/2603.05698) |

### Q: What does BEACON-SP improve, and what is it compared with?

**Answer:** BEACON-SP improves completeness, clinical relevance, and evidence grounding, and it is compared with vector-based retrieval-augmented generation (RAG) baseline.

- Grounding: cites 1 of 9 evidence facts (1 direct facts, not just 'paper mentions X'); 1 of 1 sentences have no citation
- Plan: First, find information about BEACON-SP. Then, identify what it improves. Finally, determine what it is compared with.
- Mentions found in question: BEACON-SP, improves, compared with
- Start nodes in graph: BEACON-SP
- Routing (routed by relation words in the question: improves, comparedWith): BEACON-SP --comparedWith--> Graph Retrieval-Augmented Generation (paper wording: "compared with"); BEACON-SP --comparedWith--> vector-based retrieval-augmented generation (RAG) baseline (paper wording: "compared with"); BEACON-SP --improves--> completeness, clinical relevance, and evidence grounding (paper wording: "improves"); BEACON-SP --improves--> factual accuracy (paper wording: "improves")
- Hop 1: from BEACON-SP (7 edges seen), followed: BEACON-SP --addresses--> multi-hop reasoning across diagnoses, medications, risk and protective factors, life events, and temporal relationships (paper wording: "supports"); BEACON-SP --proposes--> ontology-grounded patient knowledge graphs (paper wording: "constructed")
- Hop 2: from multi-hop reasoning across diagnoses, medications, risk and protective factors, life events, and temporal relationships, ontology-grounded patient knowledge graphs (2 edges seen), followed: BEACON-SP: Ontology-Grounded GraphRAG Framework for Clinical Suicide Risk Assessment --mentions--> multi-hop reasoning across diagnoses, medications, risk and protective factors, life events, and temporal relationships
- Hop 3: from BEACON-SP: Ontology-Grounded GraphRAG Framework for Clinical Suicide Risk Assessment (5 edges seen), followed: BEACON-SP: Ontology-Grounded GraphRAG Framework for Clinical Suicide Risk Assessment --mentions--> BEACON-SP; BEACON-SP: Ontology-Grounded GraphRAG Framework for Clinical Suicide Risk Assessment --mentions--> Suicide Social Determinants of Health Ontology

| Evidence | Fact | Source papers |
|---|---|---|
| E1 | BEACON-SP --comparedWith--> Graph Retrieval-Augmented Generation (paper wording: "compared with") | [2610.09026](https://arxiv.org/abs/2610.09026) |
| E2 | BEACON-SP --comparedWith--> vector-based retrieval-augmented generation (RAG) baseline (paper wording: "compared with") | [2610.09026](https://arxiv.org/abs/2610.09026) |
| E3 | BEACON-SP --improves--> completeness, clinical relevance, and evidence grounding (paper wording: "improves") | [2610.09026](https://arxiv.org/abs/2610.09026) |
| E4 | BEACON-SP --improves--> factual accuracy (paper wording: "improves") | [2610.09026](https://arxiv.org/abs/2610.09026) |
| E5 | BEACON-SP --addresses--> multi-hop reasoning across diagnoses, medications, risk and protective factors, life events, and temporal relationships (paper wording: "supports") | [2610.09026](https://arxiv.org/abs/2610.09026) |
| E6 | BEACON-SP --proposes--> ontology-grounded patient knowledge graphs (paper wording: "constructed") | [2610.09026](https://arxiv.org/abs/2610.09026) |
| E7 | BEACON-SP: Ontology-Grounded GraphRAG Framework for Clinical Suicide Risk Assessment --mentions--> multi-hop reasoning across diagnoses, medications, risk and protective factors, life events, and temporal relationships | [2610.09026](https://arxiv.org/abs/2610.09026) |
| E8 | BEACON-SP: Ontology-Grounded GraphRAG Framework for Clinical Suicide Risk Assessment --mentions--> BEACON-SP | [2610.09026](https://arxiv.org/abs/2610.09026) |
| E9 | BEACON-SP: Ontology-Grounded GraphRAG Framework for Clinical Suicide Risk Assessment --mentions--> Suicide Social Determinants of Health Ontology | [2610.09026](https://arxiv.org/abs/2610.09026) |

### Q: What does TagRAG compare itself with, and what does it support?

**Answer:** TagRAG compares itself with 14.6x construction efficiency, 1.9x retrieval efficiency, Graph Retrieval-Augmented Generation, and UltraDomain datasets.

- Grounding: cites 1 of 9 evidence facts (1 direct facts, not just 'paper mentions X'); 1 of 1 sentences have no citation
- Plan: Use the knowledge graph to find information about TagRAG, its comparison with other systems, and the tasks it supports.
- Mentions found in question: TagRAG, comparison with, support for
- Start nodes in graph: TagRAG
- Routing (routed by relation words in the question: comparedWith): TagRAG --comparedWith--> 14.6x construction efficiency (paper wording: "compares"); TagRAG --comparedWith--> 1.9x retrieval efficiency (paper wording: "compares"); TagRAG --comparedWith--> Graph Retrieval-Augmented Generation (paper wording: "compares"); TagRAG --comparedWith--> UltraDomain datasets (paper wording: "compares")
- Hop 1: from TagRAG (7 edges seen), followed: TagRAG --addresses--> efficient knowledge increment (paper wording: "supports"); TagRAG --proposes--> Tag-guided hierarchical knowledge graph retrieval-Augmented generation (paper wording: "proposes")
- Hop 2: from efficient knowledge increment, Tag-guided hierarchical knowledge graph retrieval-Augmented generation (2 edges seen), followed: TagRAG: Tag-guided Hierarchical Knowledge Graph Retrieval-Augmented Generation --mentions--> efficient knowledge increment
- Hop 3: from TagRAG: Tag-guided Hierarchical Knowledge Graph Retrieval-Augmented Generation (5 edges seen), followed: TagRAG: Tag-guided Hierarchical Knowledge Graph Retrieval-Augmented Generation --mentions--> 14.6x construction efficiency; TagRAG: Tag-guided Hierarchical Knowledge Graph Retrieval-Augmented Generation --mentions--> TagRAG

| Evidence | Fact | Source papers |
|---|---|---|
| E1 | TagRAG --comparedWith--> 14.6x construction efficiency (paper wording: "compares") | [2601.05254](https://arxiv.org/abs/2601.05254) |
| E2 | TagRAG --comparedWith--> 1.9x retrieval efficiency (paper wording: "compares") | [2601.05254](https://arxiv.org/abs/2601.05254) |
| E3 | TagRAG --comparedWith--> Graph Retrieval-Augmented Generation (paper wording: "compares") | [2601.05254](https://arxiv.org/abs/2601.05254) |
| E4 | TagRAG --comparedWith--> UltraDomain datasets (paper wording: "compares") | [2601.05254](https://arxiv.org/abs/2601.05254) |
| E5 | TagRAG --addresses--> efficient knowledge increment (paper wording: "supports") | [2601.05254](https://arxiv.org/abs/2601.05254) |
| E6 | TagRAG --proposes--> Tag-guided hierarchical knowledge graph retrieval-Augmented generation (paper wording: "proposes") | [2601.05254](https://arxiv.org/abs/2601.05254) |
| E7 | TagRAG: Tag-guided Hierarchical Knowledge Graph Retrieval-Augmented Generation --mentions--> efficient knowledge increment | [2601.05254](https://arxiv.org/abs/2601.05254) |
| E8 | TagRAG: Tag-guided Hierarchical Knowledge Graph Retrieval-Augmented Generation --mentions--> 14.6x construction efficiency | [2601.05254](https://arxiv.org/abs/2601.05254) |
| E9 | TagRAG: Tag-guided Hierarchical Knowledge Graph Retrieval-Augmented Generation --mentions--> TagRAG | [2601.05254](https://arxiv.org/abs/2601.05254) |
