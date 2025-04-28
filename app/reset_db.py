"""
python reset_chroma_folder.py
"""
import shutil, os, time

persist_dir = "chroma_db"

print("Stopping Django first…")   # make sure you stopped it
time.sleep(2)

if os.path.exists(persist_dir):
    shutil.rmtree(persist_dir)
    print("Deleted", persist_dir)

os.makedirs(persist_dir, exist_ok=True)
print("Fresh Chroma folder created.")





#checking the size of current db after resetting
# import chromadb
# client = chromadb.PersistentClient("chroma_db")
# print(client.list_collections())          # ==> []
# # or:
# print(sum(client.get_collection(c["name"]).count()
#           for c in client.list_collections()))    # ==> 0
