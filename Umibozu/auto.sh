# A bash loop around auto.py in case it segfaults
# See also: https://x.com/__nmca__/status/1832579590935736384

while true
do
  python "$(dirname "$0")/auto.py"
done
