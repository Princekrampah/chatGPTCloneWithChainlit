### Model Used

We'll use a quantized model. What does this mean?

#### Quantization

You are absolutely correct! Quantization is a powerful technique in the field of machine learning and deep learning that involves representing numerical data, such as weights and activations, with lower precision data types like 8-bit integers (int8) instead of the usual 32-bit floating point (float32). This technique offers several advantages, as you've mentioned:

**Reduced Memory Storage**

By using lower precision data types, the model's memory footprint is significantly reduced. This is particularly important when deploying models on memory-constrained devices or in large-scale distributed systems, where memory efficiency is crucial.

**Lower Computational Cost**

With quantization, arithmetic operations like matrix multiplication can be performed much faster using integer arithmetic. This is because integer operations are often simpler and more efficient compared to floating-point operations, which require more computational resources.

**Energy Efficiency**

In theory, reduced precision computations can lead to lower power consumption, especially when running deep learning models on hardware accelerators or resource-constrained devices. This makes quantization an attractive approach for deploying models in edge devices and embedded systems.

**Deployment on Embedded Devices**

Many embedded devices, such as smartphones, IoT devices, and microcontrollers, have hardware support for integer operations but lack efficient floating-point support. Quantization allows these models to be efficiently deployed on such devices without the need for expensive floating-point hardware.

Despite these advantages, quantization is not without its challenges. When reducing the precision of weights and activations, there may be a loss of model accuracy due to information loss. Additionally, the quantization process can introduce quantization errors, which might accumulate during the model's forward and backward passes, impacting the final performance.

To address these challenges, various techniques like post-training quantization, quantization-aware training, and mixed-precision training have been developed to optimize the quantization process and minimize the impact on model performance.

In summary, quantization is a valuable tool for optimizing and deploying deep learning models, especially in resource-constrained environments, offering benefits in terms of memory efficiency, computational speed, and energy consumption.

[Read more on quantization here](https://huggingface.co/docs/optimum/concept_guides/quantization)

[Download moder here](https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML/blob/main/llama-2-7b-chat.ggmlv3.q8_0.bin)


### Transformers

Transformers, a popular library developed by Hugging Face, provides APIs and tools that make it easy to download, use, and fine-tune state-of-the-art pretrained models based on transformer architectures. These pretrained models have been pre-trained on large-scale datasets, enabling them to capture rich patterns and semantic representations from the data.

[Read more on transformer here](https://huggingface.co/docs/transformers/main/index)

#### C-transformers

"Python bindings for the Transformer models implemented in C/C++ using GGML library."[Source](https://github.com/marella/ctransformers)

Check git repo [here](https://github.com/marella/ctransformers)


### Sentence Transformers

"SentenceTransformers is a Python framework for state-of-the-art sentence, text and image embeddings. The initial work is described in our paper Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks."[source](https://www.sbert.net/)


### Vector Stores

A vector store, also known as a vector database or vector index, is a specialized data storage and retrieval system designed to efficiently store and retrieve high-dimensional vector data. It is commonly used in applications related to machine learning, natural language processing, computer vision, recommendation systems, and other domains where vector representations are essential.

#### Similarity Algorithms In Vector Stores

In vector stores, similarity algorithms serve a crucial role in efficiently retrieving relevant vectors based on their similarity to a given query vector. These algorithms are used for the following purposes:

**Nearest Neighbor Search**

The primary use of similarity algorithms in vector stores is to perform nearest neighbor searches. Given a query vector, the vector store can quickly identify the vectors that are most similar to the query vector based on a chosen similarity metric (e.g., cosine similarity, Euclidean distance).

**Similarity-Based Retrieval**

Vector stores use similarity algorithms to retrieve vectors that are semantically or contextually similar to a given vector. This is particularly useful in applications like recommendation systems, content-based filtering, and search engines.

**Anomaly Detection**

By comparing vectors to their neighbors, similarity algorithms can identify anomalies or outliers in the data, which can be beneficial for detecting unusual patterns or anomalies in dataset

And so much more.

#### Similarity Algorithms Used


In vector stores, similarity algorithms are used to efficiently compute the similarity between vectors and perform similarity searches. These algorithms help retrieve vectors that are most similar to a given query vector. Some of the common similarity algorithms used in vector stores include:

**Cosine Similarity**

Cosine similarity is a widely used similarity measure for vectors. It computes the cosine of the angle between two vectors and ranges from -1 (completely dissimilar) to 1 (identical). Cosine similarity is particularly effective for measuring similarity between sparse vectors.

**Euclidean Distance**

Euclidean distance is a distance metric that calculates the straight-line distance between two vectors in the space. It is also used as a similarity measure by taking the inverse of the distance. Similar vectors will have smaller Euclidean distances.

**Manhattan Distance (L1 Norm)**

Manhattan distance, also known as L1 norm, calculates the sum of absolute differences between corresponding elements of two vectors. It is another distance metric used for similarity computations.

**Jaccard Similarity**

Jaccard similarity is used for set-based data. It calculates the ratio of the size of the intersection of two sets to the size of their union. It is commonly used in recommendation systems.

**Hamming Distance**

Hamming distance is used to measure similarity between binary vectors of equal length. It calculates the number of positions at which the corresponding bits are different.

**Pearson Correlation Coefficient**

The Pearson correlation coefficient measures the linear correlation between two vectors. It ranges from -1 (perfectly negatively correlated) to 1 (perfectly positively correlated).

#### Faiss-CPU

There are many vector store databases we can use, I'll use faiss cpu in this case

"Faiss is a library for efficient similarity search and clustering of dense vectors. It contains algorithms that search in sets of vectors of any size, up to ones that possibly do not fit in RAM. It also contains supporting code for evaluation and parameter tuning. Faiss is written in C++ with complete wrappers for Python/numpy. It is developed by Facebook AI Research."[Source](https://pypi.org/project/faiss-cpu/)


### Chainlit

[Chainlit documentation](https://github.com/Chainlit/chainlit)

Production Build Solutions

[Async chain/ openAI chat calls](https://github.com/langchain-ai/langchain/issues/1372)
