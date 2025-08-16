
import itertools
import json
import logging
from datetime import datetime
import pandas
import boto3
import os
import urllib.parse
import io
import csv
import random

from airflow.decorators import dag, task
from airflow.providers.amazon.aws.hooks.s3 import S3Hook
from airflow.providers.http.hooks.http import HttpHook
from airflow.datasets import Dataset

# Define connection IDs used in the Airflow UI
API_CONN_ID = "clash_royale_api"
MINIO_CONN_ID = "minio_s3"

# Define the MinIO bucket and the output filename
MINIO_BUCKET = "datasets"
DATASET_NAME="clash-royale"
DATETIME_STRING=datetime.now().strftime('%d-%m-%Y')
OUTPUT_DATASET_FILENAME = f"enriched_data.csv"
OUTPUT_DICT_FILENAME = f"number_to_card.csv"

log = logging.getLogger(__name__)

dict_dataset= Dataset(f's3://datasets/{DATASET_NAME}/{OUTPUT_DICT_FILENAME}')
enriched_data_dataset= Dataset(f's3://datasets/{DATASET_NAME}/{OUTPUT_DATASET_FILENAME}')

@dag(
    dag_id="api_to_minio_enrichment_dag",
    start_date=datetime(2025, 8, 10),
    schedule=None,
    catchup=False,
    tags=["api", "minio", "kubernetes", "taskflow"],
    doc_md="""
    ### API to MinIO DAG

    This DAG is a complete ETL pipeline:
    1.  **Extract**: Gets data from an ClashRoyale API.
    2.  **Transform**: Gets all combinations of decks .
    3.  **Load**: Saves the enriched data to an on-prem MinIO bucket.
    """
)
def api_to_minio_enrichment_dag():
    """
    DAG to fetch, enrich, and store data from ClashRoyale API to MinIO.
    """

    @task
    def get_decks_from_api() -> tuple[list[list],dict]:
        """
        Task 1: Fetches decks from latest games of 1000 best players in ClashRoyale.
        """
        log.info(f"Fetching data using HTTP connection '{API_CONN_ID}'")
        http_hook = HttpHook(method="GET", http_conn_id=API_CONN_ID)
        
        # get all cards and create id to index map
        cards_responce = http_hook.run(endpoint="v1/cards")
        names = [item['name'] for item in cards_responce.json()["items"]]
        card_name_to_number=dict(zip(names, range(len(names))))
        card_name_to_number["<Start of deck>"]=len(card_name_to_number)

        # get latest season
        seasons_responce = http_hook.run(endpoint="v1/locations/global/seasonsV2")    
        
        last_season_id = seasons_responce.json()["items"][-1]["code"]
        log.info(f"Successfully fetched last_season_id: {last_season_id}.")

        # get best players
        players_responce = http_hook.run(endpoint=f"/v1/locations/global/pathoflegend/{last_season_id}/rankings/players?limit=1000")

        best_players= players_responce.json()["items"]

        # get decks for each best player
        all_decks=[]
        for player in best_players:
            encoded_tag = urllib.parse.quote_plus(player['tag'])
            battle_log_responce= http_hook.run(endpoint=f"/v1/players/{encoded_tag}/battlelog")
            battle_log=  battle_log_responce.json()

            for battle in battle_log:
                # only from ranked 1v1 games
                if battle["type"]!="pathOfLegend":
                    continue
                team_deck_numbers = []
                opponent_deck_numbers = []

                team_cards = battle["team"][0]["cards"]
                for card in team_cards:
                    card_name = card["name"]
                    # Look up the number in the dictionary
                    if card_name in card_name_to_number:
                        card_number = card_name_to_number[card_name]
                        team_deck_numbers.append(card_number)
                    else:
                        # Handle cases where a card might not be in your mapping dictionary
                        print(f"Warning: Team card '{card_name}' not found in mapping dictionary.")
                
                opponent_cards = battle["opponent"][0]["cards"]
                for card in opponent_cards:
                    card_name = card["name"]
                    # Look up the number in the dictionary
                    if card_name in card_name_to_number:
                        card_number = card_name_to_number[card_name]
                        opponent_deck_numbers.append(card_number)
                    else:
                        # Handle cases where a card might not be in your mapping dictionary
                        print(f"Warning: Opponent card '{card_name}' not found in mapping dictionary.")
                
                opponent_deck_numbers.sort()
                team_deck_numbers.sort()
                all_decks.append(opponent_deck_numbers)
                all_decks.append(team_deck_numbers)                

        unique_decks_tuple = set(tuple(deck) for deck in all_decks)
        unique_decks_list = [list(deck) for deck in unique_decks_tuple]
        return (unique_decks_list,card_name_to_number)
 
            


    @task(outlets=[dict_dataset,enriched_data_dataset])
    def enrich_data(raw_data: tuple[list[list],dict]):
        """
        Task 2: Enriches the data by adding a new column with the processing timestamp.
        """
        data=raw_data[0]
        card_name_to_number=raw_data[1]
        if not data:
            log.warning("No data received for enrichment.")
            return []
            
        log.info(f"Enriching {len(data)} records.")
        
        # Create a new list with the enriched data
        enriched_list = []
        for row in data:
            for length in range(2, len(row) + 1):
                combinations_for_length = itertools.combinations(row, length)
                # add start of deck and shuffle every array
                for arr in combinations_for_length:
                    arr=list(arr)
                    random.shuffle(arr)
                    enriched_list.append([card_name_to_number["<Start of deck>"]]*(8-length)+arr)

            
        log.info("Enrichment complete.")


        log.info(f"Connecting to MinIO using connection '{MINIO_CONN_ID}'")

        s3_hook = S3Hook(aws_conn_id=MINIO_CONN_ID)
        
        # Save dataset
        csv_dataset_buffer = io.StringIO()
        csv_dataset_writer = csv.writer(csv_dataset_buffer)

        csv_dataset_writer.writerows(enriched_list)

        csv_dataset_string = csv_dataset_buffer.getvalue()
        
        log.info(f"Uploading data to bucket '{MINIO_BUCKET}/{DATASET_NAME}/{DATETIME_STRING}' with key '{OUTPUT_DATASET_FILENAME}'")
        s3_hook.load_string(
            string_data=csv_dataset_string,
            key=f"{DATASET_NAME}/{DATETIME_STRING}/{OUTPUT_DATASET_FILENAME}",
            bucket_name=MINIO_BUCKET,
            replace=True  # Overwrite the file if it already exists
        )


        #save inverted number map
        card_number_to_name={f"{v}": k for k, v in card_name_to_number.items()}
        csv_dict_buffer = io.StringIO()
        csv_dict_writer = csv.writer(csv_dict_buffer)

        csv_dict_writer.writerow(card_number_to_name.keys())
        csv_dict_writer.writerow(card_number_to_name.values())

        csv_dict_string = csv_dict_buffer.getvalue()
        
        log.info(f"Uploading data to bucket '{MINIO_BUCKET}/{DATASET_NAME}/{DATETIME_STRING}' with key '{OUTPUT_DICT_FILENAME}'")
        s3_hook.load_string(
            string_data=csv_dict_string,
            key=f"{DATASET_NAME}/{DATETIME_STRING}/{OUTPUT_DICT_FILENAME}",
            bucket_name=MINIO_BUCKET,
            replace=True  # Overwrite the file if it already exists
        )



    # Define the task dependencies using the TaskFlow API
    raw_data = get_decks_from_api()
    enrich_data(raw_data)


# Instantiate the DAG
dag=api_to_minio_enrichment_dag()

if __name__ == "__main__":
    current_file_path = os.path.abspath(__file__)
    parent_directory = os.path.dirname(current_file_path)
    dag.test(
        conn_file_path=os.path.join(parent_directory,"..","include","connections.yaml")
    )