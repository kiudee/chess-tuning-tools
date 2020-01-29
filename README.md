# chess-tuning-tools

## Starting the tuning client
In order to be able to start the tuning client, first create a python
environment with the packages `psycopg2` and `numpy`. 
Using anaconda this could be done by typing: 

```
conda create -n tuning -c conda-forge numpy psycopg2
```

Then after extracting the current `.zip` package into the folder
`chess-tuning-tools`, make sure that you have the following directory
structure:
```
chess-tuning-tools/
|---- networks/
|     |---- 58613
|     |---- other networks
|---- openings/
|     |---- ... 
|     |---- openings-6ply-1000.pgn
|     |---- ...
|---- tune/
|     |---- db_workers/
|     |     |---- __init__.py
|     |     |---- tuning_client.py
|     |     |---- tuning_server.py
|     |     |---- utils.py
|     |---- __init__.py
|     |---- io.py
|---- dbconfig.json
|---- lc0[.exe]
|---- sf[.exe]
```

Finally, the tuning client can be started as follows:
```
cd path/to/chess-tuning-tools
conda activate tuning
python -m tune.db_workers.tuning_client
```
