from werkzeug.security import generate_password_hash

password = 'AdminDoctor2025'
hashed_password = generate_password_hash(password)
print(hashed_password)

