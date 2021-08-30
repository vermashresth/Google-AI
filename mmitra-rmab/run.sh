USERNAME=$1
PASSWORD=$2
DATE=$3
INTERVENTIONS=$4

mkdir data
mkdir data/beneficiary
mkdir data/call
./get_data.sh ${USERNAME} ${PASSWORD} ${DATE}
pip install -r requirements.txt
unzip policy.zip
python rmab_individual_clustering.py ${DATE} ${INTERVENTIONS}
