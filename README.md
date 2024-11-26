# **Lero: A Learning-to-Rank Query Optimizer**

This repository provides a demo implementation of **Lero**, a machine learning-based query optimizer that uses a pairwise learning-to-rank approach to improve query execution plans in PostgreSQL.

## **Overview**
The Lero optimizer improves database query performance by selecting better execution plans compared to the default PostgreSQL optimizer. The repository contains code to reproduce experiments as described in the original paper. This guide details the step-by-step instructions for setting up and running the project, including handling common challenges encountered during reproduction.

---

## **Prerequisites**
1. **Linux Environment:**
   - Lero requires a Linux environment to run the provided scripts and commands. Windows users should use **WSL (Windows Subsystem for Linux)** with Ubuntu or a virtual machine.

2. **Hardware Requirements:**
   - The original implementation was designed for GPU. If no GPU is available, modify the code for CPU compatibility (instructions provided below).

3. **Dependencies:**
   - Python 3.8+
   - PostgreSQL 13.1
   - Required Python packages (see below for virtual environment setup).

4. **Dataset:**
   - The original paper uses the **STATS** dataset. If unavailable, use a synthetic dataset with similar characteristics.

---

## **Setup Instructions**

### **Step 1: Install and Configure WSL or Linux**
1. For Windows users, enable WSL and install Ubuntu:
   ```bash
   wsl --install -d Ubuntu
   ```
2. For other users, use a virtual machine with Ubuntu 20.04 or higher.

---

### **Step 2: Install PostgreSQL 13.1**
1. Download PostgreSQL 13.1 source code:
   ```bash
   wget https://ftp.postgresql.org/pub/source/v13.1/postgresql-13.1.tar.bz2
   tar -xvf postgresql-13.1.tar.bz2
   cd postgresql-13.1
   ```
2. Apply the Lero patch:
   ```bash
   git apply ../0001-init-lero.patch
   ```
3. Compile and install PostgreSQL:
   ```bash
   ./configure
   make
   sudo make install
   ```
4. Initialize the PostgreSQL database:
   ```bash
   mkdir /usr/local/pgsql/data
   /usr/local/pgsql/bin/initdb -D /usr/local/pgsql/data
   ```

5. Configure `postgresql.conf`:
   - Open `/usr/local/pgsql/data/postgresql.conf` and modify:
     ```plaintext
     listen_addresses = '*'
     geqo = off
     max_parallel_workers = 0
     max_parallel_workers_per_gather = 0
     ```
6. Start the PostgreSQL server:
   ```bash
   /usr/local/pgsql/bin/pg_ctl -D /usr/local/pgsql/data start
   ```

---

### **Step 3: Resolve Missing PostgreSQL Role**
1. If the `postgres` role does not exist, log in as superuser:
   ```bash
   sudo -u postgres /usr/local/pgsql/bin/psql
   ```
2. Create the `postgres` role:
   ```sql
   CREATE ROLE postgres WITH SUPERUSER CREATEDB CREATEROLE LOGIN PASSWORD 'yourpassword';
   ```
3. Create the required database:
   ```sql
   CREATE DATABASE stats;
   ```

---

### **Step 4: Python Environment Setup**
1. Create a virtual environment:
   ```bash
   python3 -m venv myenv
   source myenv/bin/activate
   ```
2. Install required Python packages:
   - The repository does not include a `requirements.txt` file. Use the following:
     ```bash
     pip install psycopg2 numpy torch joblib matplotlib
     ```

   - For systems without GPUs:
     ```bash
     pip install torch==1.13.0+cpu torchvision==0.14.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
     ```

---

### **Step 5: Modifying Code for CPU (If Necessary)**
If running on a CPU-only system, modify the code to disable GPU:
1. Locate GPU-dependent code in `model.py`, `server.py`, and other files.
2. Replace all `cuda()` calls with `cpu()`. For example:
   ```python
   device = torch.device("cpu")
   ```

---

### **Step 6: Start the Lero Server**
1. Open the `server.conf` file and update the model path:
   ```plaintext
   ModelPath = ./reproduce/stats_pw
   ```
2. Start the server:
   ```bash
   python server.py
   ```

---

### **Step 7: Run the Training and Testing**
1. Execute the training script:
   ```bash
   python train_model.py --query_path ../reproduce/training_query/stats.txt \
   --test_query_path ../reproduce/test_query/stats.txt --algo lero \
   --query_num_per_chunk 20 --output_query_latency_file lero_stats.log \
   --model_prefix stats_test_model --topK 3
   ```

2. After training, visualize the results using `visualization.ipynb` in Jupyter Notebook.

---

## **Common Challenges and Solutions**

### 1. **Server Fails to Start (Permission Denied)**
   - Ensure the `/usr/local/pgsql/data` directory has the correct permissions:
     ```bash
     sudo chown -R postgres:postgres /usr/local/pgsql/data
     ```

### 2. **Role or Database Missing**
   - Create the required role and database as shown in Step 3.

### 3. **GPU Dependency Errors**
   - Replace GPU-related code as detailed in Step 5.

### 4. **PostgreSQL Configuration Issues**
   - Double-check the `postgresql.conf` file for syntax errors.
   - Restart the server to apply changes:
     ```bash
     /usr/local/pgsql/bin/pg_ctl -D /usr/local/pgsql/data restart
     ```

---

## **Results**
1. **Training Performance:**
   - Compare the training time between PostgreSQL and Lero for 1000 queries (see `lero_stats.log`).

2. **Testing Performance:**
   - Evaluate the model's performance on test queries.

---

## **Acknowledgments**
This implementation is based on the paper **"Lero: A Learning-to-Rank Query Optimizer"**. For further details, refer to the paper in this repository. 

## Additional Resources
[Lero: A Learning-to-Rank Query Optimizer Paper](https://arxiv.org/abs/2302.06873)
[Lero GitHub Repo](https://github.com/AlibabaIncubator/Lero-on-PostgreSQL)