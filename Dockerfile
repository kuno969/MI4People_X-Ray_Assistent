FROM ubuntu:22.04

# Install dependencies
RUN apt update
RUN apt install python3 -y
RUN apt install python3-pip -y

# Python libraries
COPY requirements.txt .
RUN pip3 install -r requirements.txt

# Copy project files
COPY . /mnt/code/.

# Execution
WORKDIR /mnt/code
RUN chmod a+rwx run.sh
ENTRYPOINT ["./run.sh"]
