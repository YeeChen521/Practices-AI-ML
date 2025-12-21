Model Optimization: Fine-tuned an ALBERT (A Lite BERT) architecture to reduce model size by ~80% compared to traditional BERT while maintaining 9X% accuracy in identifying cross-platform spam.

Engineering Efficiency: Leveraged cross-layer parameter sharing and embedding factorization (native to ALBERT) to decrease memory footprint and increase inference speed for real-time comment filtering.

Data Augmentation: Addressed data scarcity by implementing Transfer Learning, training the model on the YouTube Spam Collection dataset and evaluating performance on scraped, unlabelled Instagram data.