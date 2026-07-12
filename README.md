#  Attention Mechanism Visualizer

Ever wondered what a Transformer is *actually* looking at when it reads a sentence? 

I built this interactive visualizer to crack open the "black box" of the Attention Mechanism. I love diving into the heavy math, system design, and AI architectures, but I absolutely despise writing frontend code or debugging web servers. So, I built this entirely in Python to run natively in Google Colab. It takes a raw sentence and visually maps out exactly how the Query, Key, and Value matrices interact.

<img width="1515" height="231" alt="image" src="https://github.com/user-attachments/assets/98ce909c-1d75-4a19-aa89-1e945334a717" />


##  The Tech Stack

*   **Language:** Python
*   **Deep Learning:** PyTorch & Hugging Face (`bert-base-uncased`)
*   **UI Framework:** Gradio (Because it just works natively in notebooks)
*   **Data Viz:** Plotly Express & Graph Objects

---

##  What This Project Actually Does

When you type a sentence into the app, it doesn't just generate text. It extracts the raw mathematical relationships between the words. 

1.  **Tokenization:** Splits your sentence into chunks (handling sub-words automatically via BERT).
2.  **Matrix Extraction:** Pulls the exact Attention Weights from Layer 0, Head 0 of a pre-trained BERT model.
3.  **Heatmap Generation:** Plots a 2D matrix where the X-axis is the Key (the word being looked at) and the Y-axis is the Query (the word doing the looking).
4.  **Bipartite Graph Routing:** Draws an interactive neural network graph showing the exact flow of attention using curved edges and dynamic node sizes.

<img width="1472" height="458" alt="image" src="https://github.com/user-attachments/assets/2fa4662c-7e62-43ff-98ae-24e2d4350965" />
 <img width="1416" height="499" alt="image" src="https://github.com/user-attachments/assets/a3fcac0e-0d77-4168-bc44-0fb86e3fc93f" />


---

##  The Bottlenecks (And What I Learned)

Building this wasn't a straight line. Here are a few major roadblocks I hit and how I solved them:

### 1. The Frontend Nightmare
Originally, I tried building this using Streamlit and exposing it via Localtunnel. It was a massive headache. Localtunnel kept dropping background connections, throwing JavaScript module errors, and breaking the dynamic imports. Since I am not fond of frontend debugging, I completely ripped out Streamlit and rewrote the UI in **Gradio**. Gradio renders directly inside the Colab cell—no tunnels, no port-forwarding, zero web dev friction. 

### 2. The "Untrained" Math Illusion
At first, I wrote the entire architecture from scratch using pure NumPy and random weights to simulate the $Q$, $K$, and $V$ matrices. The math for the formula was flawless:

$$Attention(Q, K, V) = softmax\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

But the visual output was complete gibberish! Words were paying 90% attention to random punctuation marks. It hit me that an *untrained* model is literally just random static. To make the visualizer actually useful for understanding semantic grammar, I integrated the Hugging Face API to load pre-trained weights from BERT.

### 3. Graphing Attention vs. Graphing MLPs
I initially expected the neural network graph to look like a standard Multi-Layer Perceptron (dense sequential layers). But attention doesn't work like that. It's a routing system where tokens look at *each other* in the same sequence. I had to custom-build a bipartite graph using Plotly where the Queries sit on the left, the Keys sit on the right, and the edge thickness is driven by the attention weight.

<img width="1265" height="453" alt="image" src="https://github.com/user-attachments/assets/4456347c-4dad-4116-b6c6-7ff6b80480b6" />


---

##  How to Run It Yourself

Anyone can run this in 2 minutes. You don't need to install anything on your local machine.

1.  Open the `Attention_Visualizer.ipynb` file in Google Colab.
2.  Run **Cell 1** to install Gradio and Transformers.
3.  Run **Cell 2** to load the Hugging Face BERT model into memory.
4.  Run **Cell 3** to initialize the Plotly graphing engine.
5.  Run **Cell 4** to launch the Gradio UI directly below the code block.
6.  Type a sentence and watch the math unfold!

---
*Built as a deep-dive into AI architecture and interpretability.*
