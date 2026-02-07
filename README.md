# Learning Probability Density Functions using GANs

### **Project Overview**
This project implements a **Generative Adversarial Network (GAN)** to learn and approximate the unknown Probability Density Function (PDF) of a transformed random variable. Unlike traditional statistical methods that assume a fixed distribution (like Gaussian), this approach uses the generative capabilities of GANs to model the distribution implicitly from data samples.

The feature used is the **NO2 concentration** from the India Air Quality dataset, transformed using a non-linear sine-based function specific to the student's roll number.

---

### **Step 1: Transformation Parameters**
Based on the University Roll Number **102317089**, the transformation parameters are calculated as follows:

* **Roll Number (r):** 102317089
* **Parameter $a_r$:** $0.5 \times (102317089 \pmod 7) = 0.5 \times 3 = \mathbf{1.5}$
* **Parameter $b_r$:** $0.3 \times (102317089 \pmod 5 + 1) = 0.3 \times (4 + 1) = \mathbf{1.5}$

**Transformation Equation:**
$$z = x + 1.5 \sin(1.5x)$$
*(Where $x$ is the raw NO2 concentration feature)*

---

### **Step 2: GAN Architecture**
The model consists of two competing neural networks:

#### **1. Generator (G)**
* **Input:** Random noise $z_{error} \sim \mathcal{N}(0,1)$.
* **Structure:** * Input Layer (1 unit)
    * Hidden Layer 1 (64 units) + LeakyReLU
    * Hidden Layer 2 (64 units) + LeakyReLU
    * Output Layer (1 unit, Linear activation)
* **Objective:** Transform noise into samples that follow the distribution of the real data $z$.

#### **2. Discriminator (D)**
* **Input:** Real samples $(z)$ or generated fake samples $(z_f)$.
* **Structure:** * Input Layer (1 unit)
    * Hidden Layer 1 (64 units) + LeakyReLU
    * Hidden Layer 2 (64 units) + LeakyReLU
    * Output Layer (1 unit) + Sigmoid activation
* **Objective:** Distinguish between real data and the samples produced by the Generator.



---

### **Step 3: Implementation Details**
* **Optimization:** Both networks use the **Adam Optimizer** with a learning rate of $0.0002$.
* **Loss Function:** Binary Cross Entropy (BCE) Loss.
* **Training Process:** The Discriminator is trained to maximize the probability of assigning the correct label to both real and fake samples, while the Generator is trained to minimize $\log(1 - D(G(error)))$.
* **Density Estimation:** After training, 5,000 samples are generated. The final PDF is approximated using **Kernel Density Estimation (KDE)**.

---

### **Step 4: Observations**
1.  **Mode Coverage:** The GAN successfully captures the primary density peaks of the transformed NO2 data. It manages the multi-modal nature caused by the sine transformation without "collapsing" to a single point.
2.  **Training Stability:** The use of LeakyReLU and a balanced learning rate prevented the Discriminator from overpowering the Generator early in training.
3.  **Distribution Quality:** The PDF plot shows a high degree of overlap between the real transformed data and the generator's output, indicating that the GAN implicitly learned the underlying PDF without any parametric assumptions.

---

### **How to Run**
1.  **Environment:** Ensure you have `torch`, `pandas`, `numpy`, `matplotlib`, and `seaborn` installed.
2.  **Data:** Place the `india-air-quality-data.csv` in the project directory.
3.  **Run:** Execute the Python script/notebook. The script will calculate parameters, transform the data, train the GAN for 5,000 epochs, and display the final PDF comparison.
