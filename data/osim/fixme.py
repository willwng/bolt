import re

def main():
    with open('model_motor_arms_no_hand_full_contact.osim', 'r') as f:
        content = f.read()

    # Pattern matches <orientation> followed by three space-separated numbers
    pattern = r'<orientation>(\S+)\s+(\S+)\s+(\S+)</orientation>'
    # \3 is the 3rd group, \2 is the 2nd, \1 is the 1st
    replaced_content = re.sub(pattern, r'<orientation>\3 \2 \1</orientation>', content)

    with open('model_motor_arms_no_hand_full_contact.osim', 'w') as f:
        f.write(replaced_content)

if __name__ == "__main__":
    main()