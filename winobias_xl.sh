python winobias_attn_intervention.py --gpt2-version gpt2-xl --do-filter True --split dev &&
python winobias_attn_intervention.py --gpt2-version gpt2-xl --do-filter True --split test &&
python winobias_attn_intervention.py --gpt2-version gpt2-xl --do-filter False --split dev &&
python winobias_attn_intervention.py --gpt2-version gpt2-xl --do-filter False --split test