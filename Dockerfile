# Use Python >= 3.11 for contourpy==1.3.3
FROM python:3.11-slim  

# Set the working directory in the container
WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Upgrade pip to avoid installation issues
RUN pip install --upgrade pip

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . . 

# Set the working directory to where main.py is located
WORKDIR /app/src

# Command to run the application
CMD ["python", "main.py"]
