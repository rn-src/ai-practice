### Quick and dirty usage...

First do the setup
```
pipenv install
pipenv shell
```

Now you can do some training, this might take awhile.
```
./gpt.py --model-name sherlock books/*.utf-8 --training-batch-size=5
108.txt.utf-8 - THE RETURN OF SHERLOCK H epoch 1 🔪🐟🐟🐟🐟🐟🐟🐟🐟🐟🐟🐟🐟🐟🐟🐟🐟🐟🐟🐟🐟🐟🐟🐟🐟🐟   0.730% (  1440 / 197355) loss =  3.615
```

After you train, try entering a prompt for completion
```
./gpt.py --model-name sherlock --prompt "He's dead Jim."  --prompt-output-length=10
```
