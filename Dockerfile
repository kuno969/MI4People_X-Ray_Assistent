FROM ubuntu:22.04

# Install dependencies
RUN apt update
RUN apt install python3 -y
RUN apt install python3-pip -y

# Python libraries
RUN pip3 install torch torchvision torchaudio
RUN pip3 install torchxrayvision
RUN pip3 install torchcam
RUN pip3 install streamlit

# Copy project files
COPY . /mnt/code/.

# Execution
WORKDIR /mnt/code
RUN chmod a+rwx run.sh
ENTRYPOINT ["./run.sh"]
