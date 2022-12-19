./run.sh flan-t5 codah codah
./run.sh flan-t5 tomi tomi
./run.sh flan-t5 scruples-dilemma dilemma
./run.sh flan-t5 socialiqa socialiqa
./run.sh flan-t5 social-chem social-chem

echo "remove flan-t5 cache"
rm -rf ~/.cache/huggingface/

./run.sh t0pp codah codah
./run.sh t0pp tomi tomi
./run.sh t0pp scruples-dilemma dilemma
./run.sh t0pp socialiqa socialiqa
./run.sh t0pp social-chem social-chem

echo "remove topp cache"
rm -rf ~/.cache/huggingface/

./run.sh macaw-11b codah codah