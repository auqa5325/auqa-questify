import boto3, botocore
sess = boto3.session.Session()
creds = sess.get_credentials()
frozen = creds.get_frozen_credentials() if creds else None
print("Access key:", getattr(frozen, 'access_key', None))
print("Secret present:", bool(getattr(frozen, 'secret_key', None)))
print("Token present:", bool(getattr(frozen, 'token', None)))
print("Region:", sess.region_name)
