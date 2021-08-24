USERNAME=$1
PASSWORD=$2
DATE=$3

## Yet to finalise, clarifications need from online database 

# Get Beneficiary Data
echo "SELECT u.id user_id, MD5(phone_no) phone_no, lmp, enroll_gest_age, u.project_id, manager_id, call_slots, enroll_delivery_status,
delivery, LANGUAGE, registration_date, delivery_date, entry_date, deleted, phone_type, phone_code,
stage, channel_id, CASE c.channel_type WHEN 1 THEN 'Community' WHEN 2 THEN 'Hospital' ELSE 'ARMMAN' END AS ChannelType,
updated_time, ngo_hosp_id, org_type, unique_sub_id, moved, delete_reason, delete_other_reason, entry_madeby, entry_updatedby,
force_delivery_updated, completed,  dnd_optout_status, updated_status_on, age, education, enroll_phone_owner, phone_owner,
MD5(alternate_no) alternate_no, alternate_no_owner, birth_place, name_of_sakhi, name_of_project_officer, income_bracket, data_entry_officer,
g, p, s, l, a, ppc_bloodpressure, ppc_diabetes, ppc_cesarean, ppc_thyroid,  ppc_fibroid, ppc_spontaneousAbortion, ppc_heightLess140,
ppc_pretermDelivery, ppc_anaemia, ppc_otherComplications, name_of_medication_any, planned_place_of_delivery, registered_where, registered_pregnancy,
place_of_delivery, type_of_delivery, date_registration_hospital, term_of_delivery, medication_after_delivery
FROM vw_beneficiaries u
LEFT OUTER JOIN call_slot csl ON csl.id = u.call_slots
LEFT OUTER JOIN channels c ON c.id = u.ngo_hosp_id
ORDER BY u.id;" | /google/data/ro/projects/speckle/mysql -h 34.93.230.87 -P 3306 -u ${USERNAME} --password=${PASSWORD} mmitrav2 > data/beneficiary/beneficiary_pilot_data.csv
sed -i 's/LANGUAGE/language/' data/beneficiary/beneficiary_pilot_data.csv

# Call data
echo "SELECT id, user_id, startdatetime, enddatetime, duration, gest_age, dropreason, callStatus, missedcall_id, media_id, esb_trans_id, tid, tsp_id
FROM vw_call_logs;" | /google/data/ro/projects/speckle/mysql -h 34.93.237.61 -P 3306 -u ${USERNAME} --password=${PASSWORD} mmitrav2 > data/call_data.csv

# Intervention lists
echo "SELECT beneficiary_id, intervention_date
FROM vw_intervention_list
WHERE intervention_date < ${DATE} AND intervention_date >= ${DATE}-21;"| /google/data/ro/projects/speckle/mysql -h 34.93.237.61 -P 3306 -u ${USERNAME} --password=${PASSWORD} mmitrav2 > data/intervention_data.csv

