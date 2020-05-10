# based on tutorial from http://boto3.readthedocs.io/en/latest/guide/dynamodb.html
# before running this code, set up AWS credentials as per instructions in
# readme file at https://github.com/boto/boto3
# this lets you create and populate a DynamoDB database from you local machine or an EC2 instance

import boto3
import time

start = time.time()

# get the service resource
dynamodb = boto3.resource('dynamodb')

# create the DynamoDB table - used if there is no such table
table = dynamodb.create_table(
    TableName='CHAT2',
    KeySchema=[ {'AttributeName': 'Product', 'KeyType': 'HASH'} ],
    AttributeDefinitions=[ {'AttributeName': 'Product', 'AttributeType': 'S'}, ],
    ProvisionedThroughput={'ReadCapacityUnits': 10, 'WriteCapacityUnits': 10})

# wait until the table exists.
table.meta.client.get_waiter('table_exists').wait(TableName='CHAT2')

table = dynamodb.Table('CHAT2')
count = 0

# open csv file with data
with open('results_entire_dataset.txt', 'r') as csvfile:

    try:

        # split line
        for line in csvfile:
            count += 1
            line = line.decode('utf-8', 'ignore')
            line = line.strip().split(';')

            # DynamoDB does not accept any empty elements
            for i in range(len(line)):
                if (len(line[i]) == 0) or (line[i] == ' '):
                    line[i] = 'None'
                line[i] = line[i].strip()

            # add items to DynamoDB table
            table.put_item(
               Item={
                                'Product': line[0],
                            'Ingredients': line[1],
                             'Reactivity': line[2],
                    'Conditions_to_avoid': line[3],
                                    'PPE': line[4],
                }
            )
            if count % 100 == 0:
                print(str(count) + " lines added")

    except Exception:
        pass

end = time.time()
diff = end - start

if diff > 3600:
    diff = diff / 3600
    elapsed = str(diff) + ' hours'
else:
    diff = diff / 60
    elapsed = str(diff) + ' min'

print('Transfer completed!')
print('Time elapsed: ' + elapsed)