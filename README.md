### installation
```
virtualenv -p python3.7 venv
cd venv
source bin/activate
pip install --upgrade pip
pip install rasa lime Flask notebook pandas
cd ..
git clone https://github.com/Naviden/XAI-Bot.git
```

### open the project in the Jupyter notebook
```
ipython kernel install  --user --name=venv
python3.7 -m notebook
```

### running the bot (terminal)
```
python xaibot.py
```

### running UI
```
cd UI
python app.py
```