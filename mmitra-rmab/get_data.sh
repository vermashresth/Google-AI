USERNAME=$1
PASSWORD=$2
DATE=$3

## Yet to finalise, clarifications need from online database 

# Get Beneficiary Data
echo "SELECT u.beneficiary_id user_id, MD5(phone_no) phone_no, lmp_date lmp, enrollment_gestation_age enroll_gest_age, u.project_id, u.call_slot_id call_slots, enrollment_delivery_status enroll_delivery_status,
language_id LANGUAGE, registration_date, delivery_date, entry_date, phone_type, phone_code, phone_owner,
u.channel_id ngo_hosp_id, CASE c.channel_type WHEN 1 THEN 'Community' WHEN 2 THEN 'Hospital' ELSE 'ARMMAN' END AS ChannelType,
unique_id unique_sub_id, entry_madeby, entry_updatedby,
forced_delivery_update force_delivery_updated, completed,  dnd_optout_status, age, education_id education,
MD5(alternate_phone_no) alternate_no, alternate_phone_owner alternate_no_owner, name_of_sakhi, name_of_project_officer, income_bracket, data_entry_officer,
g, p, s, l, a, ppc_bloodpressure, ppc_diabetes, ppc_cesarean, ppc_thyroid,  ppc_fibroid, ppc_spontaneousAbortion, ppc_heightLess140,
ppc_pretermDelivery, ppc_anaemia, ppc_otherComplications, name_of_medication_any, planned_place_of_delivery, registered_where, registered_pregnancy,
place_of_delivery, type_of_delivery, date_registration_hospital, term_of_delivery, medication_after_delivery
FROM vw_beneficiaries u
WHERE isactive = 1
LEFT OUTER JOIN call_slot csl ON csl.call_slot_id = u.call_slot_id
LEFT OUTER JOIN channels c ON c.channel_id = u.channel_id
ORDER BY u.id;" | /google/data/ro/projects/speckle/mysql -h 34.93.237.61 -P 3306 -u ${USERNAME} --password=${PASSWORD} mmitrav2 > data/beneficiary/beneficiary_pilot_data.csv
sed -i 's/LANGUAGE/language/' data/beneficiary/beneficiary_pilot_data.csv

# Call data
echo "SELECT beneficiary_id user_id, startdatetime, enddatetime, duration, gest_age, dropreason, call_status_id callStatus, missed_call_id missedcall_id, media_id, esb_trans_id, tsp_id
FROM vw_call_logs
WHERE startdatetime < ${DATE} AND startdatetime >= date_add(${DATE}, interval -7 day);" | /google/data/ro/projects/speckle/mysql -h 34.93.237.61 -P 3306 -u ${USERNAME} --password=${PASSWORD} mmitrav2 > data/call/call_data.csv

# Intervention lists
echo "SELECT beneficiary_id, intervention_date
FROM vw_intervention_list
WHERE intervention_date < ${DATE} AND intervention_date >= date_add(${DATE}, interval -21 day) AND intervention_success = 1;"| /google/data/ro/projects/speckle/mysql -h 34.93.237.61 -P 3306 -u ${USERNAME} --password=${PASSWORD} mmitrav2 > data/intervention_data.csv

