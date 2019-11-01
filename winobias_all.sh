python winobias_attn_intervention.py --gpt2-version distilgpt2 --do-filter True --split dev &&
python winobias_attn_intervention.py --gpt2-version gpt2 --do-filter True --split dev &&
python winobias_attn_intervention.py --gpt2-version gpt2-medium --do-filter True --split dev &&
python winobias_attn_intervention.py --gpt2-version gpt2-large --do-filter True --split dev &&
python winobias_attn_intervention.py --gpt2-version distilgpt2 --do-filter True --split test &&
python winobias_attn_intervention.py --gpt2-version gpt2 --do-filter True --split test &&
python winobias_attn_intervention.py --gpt2-version gpt2-medium --do-filter True --split test &&
python winobias_attn_intervention.py --gpt2-version gpt2-large --do-filter True --split test &&
python winobias_attn_intervention.py --gpt2-version distilgpt2 --do-filter False --split dev &&
python winobias_attn_intervention.py --gpt2-version gpt2 --do-filter False --split dev &&
python winobias_attn_intervention.py --gpt2-version gpt2-medium --do-filter False --split dev &&
python winobias_attn_intervention.py --gpt2-version gpt2-large --do-filter False --split dev &&
python winobias_attn_intervention.py --gpt2-version distilgpt2 --do-filter False --split test &&
python winobias_attn_intervention.py --gpt2-version gpt2 --do-filter False --split test &&
python winobias_attn_intervention.py --gpt2-version gpt2-medium --do-filter False --split test &&
python winobias_attn_intervention.py --gpt2-version gpt2-large --do-filter False --split test