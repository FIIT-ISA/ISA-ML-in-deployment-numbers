# ISA course: ML/DL deployment - Number data type

Example of LSTM sequence inference with two plot graphs, one plotting the sequences, the other showcasing predicted and real values.

Author(s):
- Matej Volansky (2024)
- as a part of the team work preparation in 2023/2024

Docker image size: `1.26 GB`


## Setup
First train your model and save the `.pt` state dictionary after training (considering you're using PyTorch).
Update the `model/model.py` with your model. Don't forget that you have to create test data for the inference the same way you do for training, so update it in `utils/sequence_dataset.py` if you change the way you generate datasets.

__*Note:*__ The technologies demostrated in this example are just selected ones. You can chose other technologies as you like (or the best you deal with).

The way sequences are plotted in this project depends on the way the data are sent between backend (`app.py`) and frontend. If you are using this example for some other sequences, change the way they are loaded and plotted, since this example is pretty specific.

If VanillaJS seems like a lot of trouble for plotting graphs for you, you can change the architecture to run with two docker images ( one for frontend, another for backend ), and in the frontend one have React run in it (for example, or any framework you prefer). Or you could even create the plots at the backend with python and then send them to frontend as images too. 

## Run
To use this inference, just run 

```
docker compose up
```
After successful build, your server will be available at `http://localhost:8080/`

For manual running without docker you have to create a python virtual environment.

```
python -m venv venv

source venv/bin/activate          # on Linux based distros
source venv\Scripts\activate.ps1  # on Windows (Powershell)
source venv\Scripts\activate.bat  # on Windows (cmd)
```

To install the PyTorch CPU version run
```
pip install torch==2.2.2+cpu torchvision==0.17.2+cpu -f https://download.pytorch.org/whl/torch_stable.html
```
After that go ahead and run
```
pip install -r requirements.txt
```
