export USERNAME=$1

echo "${USERNAME}"

if [[ $USERNAME == "mieseung" ]]; then
    git config user.name "Miseung Kim"
    git config user.email "mieseung@gmail.com"
fi
