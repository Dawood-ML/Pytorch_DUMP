# My PyTorch Mastery Journey: The Clinical Trial Method

This repository documents my journey to master Pytorch, from foundational skills to production-level deployment. I am moving beyond academic examples to build robust, professional skills using a rigorous, two-part methodology for each concept.

## Core Philosophy: The Scalpel & The Surgeon

Every new skill is developed and tested in two stages, mirroring the training of a surgeon:

1.  **The Clinical Environment (The Scalpel):** A new tool or technique is engineered with surgical precision in a controlled environment using a clean, academic dataset. The output is a piece of professional, reusable code—a perfect scalpel.
2.  **The Crucible (The Surgeon):** The exact code is then immediately applied to a messy, challenging, real-world dataset. This forces adaptation, debugging, and a deep understanding of the tool's limitations and failure modes. This is where the scalpel is wielded by a surgeon to solve a real problem.

This process builds anti-fragile skills, focusing not just on how to build things, but on how to fix them when they break.

---

## The Complete Roadmap

*Each chunk represents a skill acquired in the clinic and stress-tested in the crucible.*

### Phase 1: Foundation & Professional Workflow
*Mastering the core mechanics of PyTorch and the engineering of a professional training process.*

*   **[✅] Chunks 1-5: The Fundamentals** - Tensors, Autograd, `nn.Module`, Loss, Optimizers.
*   **[➡️] Chunk 6: Training Loop Mastery**
    *   **Clinical:** Engineer a professional, reusable `train.py` script with logging, command-line arguments, and model saving on **FashionMNIST**.
    *   **Crucible:** Stress-test the script on the **Clothing1M** dataset (subset) with ~40% noisy labels, forcing adaptation and debugging.
*   **[ ] Chunk 7: Data Pipeline Mastery (`Dataset` & `DataLoader`)**
    *   **Clinical:** Build a custom `Dataset` for a simple image folder structure (`/class_a`, `/class_b`).
    *   **Crucible:** Adapt the pipeline for a large dataset that doesn't fit in memory (e.g., a Kaggle dataset), implementing efficient transforms and optimizing `num_workers`.
*   **[ ] Chunk 8: Advanced Metrics & Evaluation**
    *   **Clinical:** Implement precision, recall, and F1-score for the balanced **CIFAR-10** dataset.
    *   **Crucible:** Tackle an imbalanced classification problem (e.g., Credit Card Fraud from Kaggle), demonstrating why accuracy is a flawed metric and implementing AUC-ROC curves.

---
### Phase 2: Architecture Mastery
*Building, training, and adapting canonical deep learning architectures for real-world domains.*

*   **[ ] Chunk 9: CNN Fundamentals**
    *   **Clinical:** Build a classic VGG-style CNN from scratch for **CIFAR-10**.
    *   **Crucible:** Apply the same CNN to a medical imaging dataset (e.g., Chest X-Rays for Pneumonia), diagnosing and mitigating the effects of domain shift.
*   **[ ] Chunk 10: Advanced CNNs & Transfer Learning**
    *   **Clinical:** Fine-tune a pre-trained ResNet-18 on the general **Caltech101** image dataset.
    *   **Crucible:** Apply transfer learning to a fine-grained classification task (e.g., **Stanford Dogs**), where pre-trained features can struggle, forcing careful layer unfreezing and tuning.
*   **[ ] Chunk 11: RNNs/LSTMs for Sequences**
    *   **Clinical:** Build an LSTM for sentiment analysis on the clean, binary **IMDb** movie reviews dataset.
    *   **Crucible:** Adapt the sequence model for a time-series forecasting problem (e.g., weather or stock data), which involves continuous data, different normalization, and a regression head.
*   **[ ] Chunk 12: Transformers & The Hugging Face Ecosystem**
    *   **Clinical:** Use a pre-trained BERT from Hugging Face for a simple text classification task (**20 Newsgroups** dataset).
    *   **Crucible:** Tackle a more complex NLP task like Question Answering on the **SQuAD** dataset, which requires a different model head and complex post-processing.
*   **[ ] Chunk 13: Autoencoders for Representation Learning**
    *   **Clinical:** Build a convolutional autoencoder to reconstruct **FashionMNIST** images and visualize the learned latent space.
    *   **Crucible:** Use the autoencoder for anomaly detection. Train it only on "normal" data (e.g., `MVTec AD` dataset) and use high reconstruction error to identify defective items.
*   **[ ] Chunk 14: Generative Adversarial Networks (GANs)**
    *   **Clinical:** Implement a simple DCGAN to generate handwritten digits from the **MNIST** dataset.
    *   **Crucible:** Attempt to train the same GAN architecture on a more complex dataset like **CelebA** (faces), encountering and debugging common GAN training issues like mode collapse.
*   **[ ] Chunk 15: Handling Multi-Modal Data**
    *   **Clinical:** Build a model that combines pre-extracted image features and tabular data for a simple prediction task.
    *   **Crucible:** Tackle a real Kaggle competition that involves combining raw text, tabular data, and images, forcing the creation of a custom multi-input architecture.

---
### Phase 3: Production, Deployment & Scale
*Bridging the gap from a trained model to a useful, scalable application.*

*   **[ ] Chunk 16: Advanced Training Techniques**
    *   **Clinical:** Implement learning rate scheduling and weight decay (regularization) on a **CIFAR-100** training run.
    *   **Crucible:** Use techniques like Gradient Clipping and Mixed-Precision Training (`torch.cuda.amp`) to successfully train a large, unstable model (like a Transformer) that might otherwise fail.
*   **[ ] Chunk 17: MLOps - Experiment Tracking**
    *   **Clinical:** Integrate **Weights & Biases** (`wandb`) into the `train.py` script to automatically log metrics, parameters, and model checkpoints.
    *   **Crucible:** Use `wandb Sweeps` to run an automated hyperparameter search to significantly improve the performance of the noisy-label model from Chunk 6.
*   **[ ] Chunk 18: Model Inference & Deployment**
    *   **Clinical:** Save a trained model and wrap it in a simple **Flask/FastAPI** web server to get predictions via an API endpoint.
    *   **Crucible:** Optimize the model for production using **TorchScript** or **ONNX Runtime**. Containerize the API using **Docker** and benchmark the latency difference.
*   **[ ] Chunk 19: Explainability & Interpretation**
    *   **Clinical:** Use `Captum` or simple gradient analysis to see which pixels the CNN from Chunk 9 looks at to make a decision on **CIFAR-10** images.
    *   **Crucible:** Apply these techniques to the medical X-ray model. Do the model's explanations align with medical knowledge, or has it learned a spurious correlation (e.g., a hospital marking)?
*   **[ ] Chunk 20: Capstone Project - The Final Crucible**
    *   There is no clinical environment here. Choose one of three challenging, real-world projects and build a complete, end-to-end solution, from data processing to a final report or deployed artifact. This is the final exam that synthesizes all learned skills.

---

## Professional Workflow

-   **Exploration & Debugging:** Jupyter Notebooks (`.ipynb`) are used as a lab for data analysis and iterative problem-solving, primarily in the Crucible phase.
-   **Core Logic & Tools:** Reusable code (models, training loops, data utilities) is engineered in Python scripts (`.py`) for modularity, version control, and automation.

## Setup
Install dependencies and set up experiment tracking:
```bash
pip install -r requirements.txt
pip install wandb
# Then login to your account
wandb login
