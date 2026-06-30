#!/bin/bash
# Copy DECA repo assets into data/ if missing (repo clone has them in data_repo/)
for f in /app/DECA/data_repo/*; do
    fname=$(basename "$f")
    if [ ! -e "/app/DECA/data/$fname" ]; then
        cp -r "$f" "/app/DECA/data/$fname"
    fi
done

# Copy models into DECA data directory
if [ -f /app/models/deca_model.tar ]; then
    cp -n /app/models/deca_model.tar /app/DECA/data/
fi
if [ -f /app/models/generic_model.pkl ]; then
    cp -n /app/models/generic_model.pkl /app/DECA/data/
fi

exec "$@"
