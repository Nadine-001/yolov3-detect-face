import os

# Fungsi untuk membuat file txt dari folder gambar
def create_file_list(folder_path, output_file):
    # Pastikan folder ada
    if not os.path.exists(folder_path):
        print(f"Folder {folder_path} tidak ditemukan.")
        return

    # Buka file output untuk ditulis
    with open(output_file, 'w') as file:
        # Iterasi melalui semua file dalam folder
        for filename in os.listdir(folder_path):
            # Filter hanya file gambar (ekstensi bisa disesuaikan)
            if filename.endswith(('.jpg', '.jpeg', '.png')):
                # Tulis path absolut file ke file txt
                file.write(os.path.abspath(os.path.join(folder_path, filename)) + '\n')

    print(f"File {output_file} berhasil dibuat.")

# Path ke folder dataset
train_folder = "8125761643/train"
val_folder = "8125761643/val"

# Nama file output
train_txt = "train_8125761643.txt"
val_txt = "val_8125761643.txt"

# Buat file train.txt dan val.txt
create_file_list(train_folder, train_txt)
create_file_list(val_folder, val_txt)
