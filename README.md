#  Attention Mechanism Visualizer: Inside the Black Box

Ever wondered what a Transformer is *actually* looking at when it reads a sentence? 

I built this interactive visualizer to crack open the "black box" of the Attention Mechanism. As an engineering student heavily focused on advanced mathematics, system design, and AI safety, I love diving into the theoretical architecture of large language models. But let's be real: I absolutely despise writing frontend code, wrestling with web frameworks, or debugging background servers. 

I wanted a tool that let me see the exact mathematical routing of a neural network without having to leave my notebook. So, I built this entirely in Python to run natively inside Google Colab. It takes a raw text input, runs it through a real pre-trained language model, and visually maps out exactly how the Query, Key, and Value matrices interact.

<img width="1689" height="453" alt="image" src="https://github.com/user-attachments/assets/9dc7a255-5759-4489-925f-0aa5ba35c0ab" />



##  The Tech Stack (Zero Frontend Required)

*   **Language:** Python
*   **Deep Learning:** PyTorch & Hugging Face (`bert-base-uncased`)
*   **UI Framework:** Gradio (Because it runs natively in Colab cells without tunneling headaches)
*   **Data Viz:** Plotly Express & Plotly Graph Objects (for interactive, hoverable visualizations)

---

##  What This Project Actually Does

When you type a sentence into the app, it doesn't generate text. It extracts the raw mathematical relationships between the words and visualizes the internal routing. Here is the exact pipeline:

1.  **Sub-word Tokenization:** It splits your sentence into chunks. If you type a word the model hasn't seen before, BERT automatically breaks it down into sub-words.
2.  **Matrix Extraction:** It bypasses the standard generation output and instead hooks directly into the network's hidden states, pulling the exact Attention Weights from a specific layer and head.
3.  **Heatmap Generation:** It plots a 2D probability matrix. The X-axis is the **Key** (the word containing information) and the Y-axis is the **Query** (the word actively looking for context). 
4.  **Bipartite Graph Routing:** It draws an interactive neural network graph showing the exact flow of attention. Node sizes scale dynamically based on how much attention they receive, and the curved edges represent the probability weights. 

---

##  Navigating the Grid: Layers and Heads

`bert-base-uncased` has 12 layers and 12 attention heads per layer. This means there are **144 different attention matrices** calculating simultaneously for every single sentence. I built interactive sliders into the UI so you can traverse this entire grid in real-time. 

Here is what happens to the neural network when you move them:

*   **The Layer Slider (Depth):** Moving this takes you deeper into the model's semantic processing pipeline. 
    *   *Early layers (0-2)* act like the model's spatial awareness. They mostly look at adjacent words to anchor themselves in the sentence. 
    *   *Middle layers (3-8)* handle basic syntax, like connecting active verbs to their subjects. 
    *   *Deep layers (9-11)* handle highly abstract meaning and complex logic. This is where the model resolves obscure pronoun references (e.g., figuring out who "her" refers to in a crowded sentence).
*   **The Head Slider (Breadth):** Inside every layer, there are 12 "Heads" calculating in parallel. Sliding through these doesn't make the output "more accurate"—it just shifts your view across different parallel specialists. One head might be entirely dedicated to finding punctuation, while another strictly tracks nouns. 

---

##  Digging Deeper: Attention Sinks & Mechanistic Interpretability

Once the visualizer was hooked up to Hugging Face, I noticed some bizarre behavior in the matrices that taught me a lot about mechanistic interpretability.

### The `[CLS]` and `[SEP]` Problem (Attention Sinks)
Initially, the right edge of my heatmap was a blindingly bright wall. Normal words were paying up to 85% of their attention to the `[SEP]` token at the end of the sentence. I thought my extraction code was completely broken, but it turns out this is a heavily documented mathematical quirk in transformer architecture. 

Because the attention equation relies on a `softmax` function, the weights of every row *must* sum to exactly $1.0$. If a specific attention head is designed to look for a direct object, but the current token is just a comma, the network still has to distribute that $1.0$ weight somewhere. Over billions of training steps, the model learns to use the `[CLS]` and `[SEP]` boundary tokens as "attention sinks" or "no-op" (no operation) routing destinations. It dumps the excess probability mass there so it doesn't contaminate the actual word context. 

While architecturally accurate, it totally skewed the visual color scale. I updated the extraction script to slice these boundary tokens out of the arrays before plotting, which immediately revealed the subtle, hidden word-to-word semantic connections.

### Reverse-Engineering the Black Box 
The most fascinating part of this project was realizing that there is no official manual detailing what each attention head does. When the model was trained, the engineers didn't assign specific grammatical tasks to them; the network organically evolved these neural circuits because dividing the labor is the mathematically optimal way to parse language.

Figuring out the purpose of individual heads is a core challenge in AI safety research. Since we don't have a map, we have to reverse-engineer the model's cognition. Researchers use statistical probing (averaging the attention of 10,000 sentences to see what a head consistently stares at) or ablation studies (forcing a head's weights to zero to see what grammatical capability the model loses). When you play with the sliders in this visualizer, you are actively hunting for the specific neural circuits that understand human language.

#### With CLS and SEP 
<img width="1470" height="429" alt="image" src="https://github.com/user-attachments/assets/3ed24fab-c760-48ae-a856-869d237ca689" />
<img width="1442" height="486" alt="image" src="https://github.com/user-attachments/assets/3b64e320-83ee-470e-83a9-2999e7007784" />

#### Without CLS and SEP 
<img width="1477" height="449" alt="image" src="https://github.com/user-attachments/assets/c509edca-1622-4bdb-94d5-1b8e9945110c" />
<img width="1324" height="484" alt="image" src="https://github.com/user-attachments/assets/00942065-582d-47a9-9ed1-baa7e771caff" />





---

##  The Bottlenecks (And What I Learned)

Building this wasn't a straight line. I ran into several massive roadblocks that forced me to rethink the architecture entirely.

### 1. The Frontend & Tunneling Nightmare
My original plan was to build the UI using Streamlit. Because I was developing in Google Colab, I had to use a tool called `localtunnel` to expose the virtual machine's local port to the public web. It was an absolute disaster. 

Localtunnel kept dropping background connections. Colab cells would hang endlessly. Every time Streamlit tried to load its dynamic JavaScript chunks in the background, the tunnel would hiccup and throw a fatal `TypeError: Failed to fetch dynamically imported module`. I spent more time hard-refreshing my browser and restarting background servers than I did writing machine learning code. Because I hate frontend debugging, I completely ripped out Streamlit and rewrote the UI in **Gradio**. Gradio renders its interface directly inside the Colab cell iframe—no tunnels, no port-forwarding, zero web dev friction. 

### 2. The "Untrained" Math Illusion
Before using Hugging Face, I tried to build the attention mechanism completely from scratch using pure NumPy and PyTorch. I initialized my own embedding layers and generated random matrices for the Query, Key, and Value weights. 

The linear algebra was flawless. I coded the exact paper implementation:
$$Attention(Q, K, V) = softmax\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

But when I rendered the heatmap, the visual output was complete gibberish. The word "in" was paying 94% of its attention to the word "saw," and nouns were heavily fixated on random punctuation marks. It hit me that an *untrained* neural network is literally just random static. Without running a backward pass over a massive dataset to optimize the $W_q$, $W_k$, and $W_v$ weights, the dot products are completely meaningless. To make the visualizer actually useful for understanding semantic grammar, I had to drop the manual NumPy approach and integrate the Hugging Face API to load actual, pre-trained weights.

### 3. Graphing Attention vs. Graphing MLPs
When you look at textbook diagrams of neural networks, they always show Multi-Layer Perceptrons (MLPs)—neat, sequential columns of nodes where layer 1 connects to layer 2, which connects to layer 3. 

I initially tried to graph my attention mechanism like this, but it fundamentally misrepresented the math. Attention doesn't work sequentially; it's a dynamic routing system where words look at *other words* within the exact same sequence. I had to custom-build a bipartite graph using Plotly where the Queries sit on the left, the Keys sit on the right, and the edge thickness is driven by the attention weight. Getting the bezier curves (curved lines) to render properly in Plotly so the graph didn't look like a chaotic web of straight overlapping lines took a lot of documentation digging.

---

##  How to Run It Yourself

Anyone can run this in 2 minutes. You don't need to install anything on your local machine.

1.  Open the `Attention_Visualizer.ipynb` file in Google Colab.
2.  Run **Cell 1** to install Gradio and Transformers.
3.  Run **Cell 2** to load the Hugging Face BERT model into memory.
4.  Run **Cell 3** to initialize the Plotly graphing engine.
5.  Run **Cell 4** to launch the Gradio UI directly below the code block.
6.  Type a sentence, move the sliders, and watch the math unfold!
