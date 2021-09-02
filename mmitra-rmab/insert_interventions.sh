#!/bin/bash
USERNAME=$1
PASSWORD=$2
DATE=$3
echo "INSERT INTO intervention_header (interventiontype_id, start_date) VALUES (2, '${DATE}');"| /google/data/ro/projects/speckle/mysql -h 34.93.237.61 -P 3306 -u ${USERNAME} --password=${PASSWORD} mmitrav2 > insertion_log.csv
while IFS=, read -r name; do
  echo "SELECT @id := intervention_id FROM intervention_header WHERE start_date='${DATE}';INSERT INTO intervention_list (intervention_id, beneficiary_id) VALUES (@id, ${name});"| /google/data/ro/projects/speckle/mysql -h 34.93.237.61 -P 3306 -u ${USERNAME} --password=${PASSWORD} mmitrav2 >> insertion_log.csv
done < user_interventions.csv
