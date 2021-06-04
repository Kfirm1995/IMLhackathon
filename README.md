IML Hackathon Course 2021

-task1:
    -memory_maps:
        - company_id_map.json - json with best correlated production companies
        - company_id_map_to_rev.json  - best correlated with revenue
        - company_id_map_to_vote.json - best correlated with vote average
        - genres.json - all genres of movies
        - top_actor_set.pickle - pickle to load of best actors
        - top_direrctor.json - best directors
        - top_original_languages.json - most freq of languages
    -models:
        folder of best models which had been trained, validated and predicted with our sample set
    -inflation_data.csv-
        csv file which includes inflation rate and usd value from 1900
    - model_eval.py:
        model evaluation for all models with plotting
    - parse.py
        cleaning data, creating featuring and loading data
    regression.py:
        main class, includes prediction 
    -utils.py:
        includes few functions which had been made generic
    -USERS
    -README
    -project.pdf
    
    
        