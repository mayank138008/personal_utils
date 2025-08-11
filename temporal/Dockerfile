# FROM python:3.10-slim

# WORKDIR /app
# COPY . /app

# RUN pip install --no-cache-dir -r requirements.txt

# ENTRYPOINT ["python", "run_one_product.py"]

# FROM python:3.10-slim

# # Install system dependencies
# RUN apt-get update && apt-get install -y \
#     g++ gcc \
#     libgdal-dev gdal-bin \
#     libgl1-mesa-glx \
#     libspatialindex-dev \
#     libxml2 libproj-dev proj-data \
#     libexpat1 \
#     && rm -rf /var/lib/apt/lists/*

# WORKDIR /app

# # Copy requirements separately to cache installs
# COPY requirements.txt .
# RUN pip install --no-cache-dir -r requirements.txt

# # Copy actual code
# COPY . /app

# CMD ["python", "run_one_product.py"]



FROM python:3.10

# Install system dependencies
RUN apt-get update && apt-get install -y \
    g++ gcc \
    libgdal-dev gdal-bin \
    libgl1-mesa-glx \
    libspatialindex-dev \
    libxml2 libproj-dev proj-data \
    libexpat1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt




# 👇 Add these lines to copy what you want
COPY .env .          
COPY config/ config/  
COPY data/ data/   
COPY models/ models/   

# Copy code
COPY . .

# Default run command
CMD ["python", "run_one_product.py"]
