#!/bin/bash
for file in *; do
    # Define the substring to remove
    substring="modelflax-sentence-embeddings_all_datasets_v4_mpnet-base_"
    
    # Create the new filename by removing the substring
    newfile=${file//$substring/}
    
    # Rename the file if the name has changed
    if [ "$file" != "$newfile" ]; then
        mv "$file" "$newfile"
    fi
done
