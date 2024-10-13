import zipfile
import os
import shutil
import json
import sqlite3
import uuid
import hashlib
from aiida.common.log import AIIDA_LOGGER

__all__ = ('CONVERSION_LOGGER', 'convert_archive_key_format')

CONVERSION_LOGGER = AIIDA_LOGGER.getChild('export')

def convert_archive_key_format(key_format_from : str, key_format_to : str, file_path : str, test_run: bool) -> str:
    """
        Converts an archive key format to a new key_format
        
        Supported conversions:
            - sha256 to uuid4
            - uuid4 to sha256
        
        :param key_format_from: archive key format
        :param key_format_to: key format the archive will be converted to
        :param file_path: path of the archive
        :param test_run: if true the archive will not be converted
            
        :returns: path of the converted archive
    """
    
    if key_format_from == key_format_to:
        return file_path
    
    CONVERSION_LOGGER.report(f"Converting archive from {key_format_from} to {key_format_to}")
    
    if key_format_from == 'sha256' and key_format_to == 'uuid4':
        if test_run:
            return file_path
        
        return _convert_archive_export(file_path, key_format_to, _sha256_to_uuid4)
    elif key_format_from == 'uuid4' and key_format_to == 'sha256':
        
        if test_run:
            return file_path
        
        return _convert_archive_export(file_path, key_format_to, _uuid4_to_sha256)
    else:
        raise NotImplementedError(f"Conversion from {key_format_from} to {key_format_to} is not supported")

def _zip_directory(folder_path, output_zip_path):
    with zipfile.ZipFile(output_zip_path, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                # Write the file into the zip archive, maintaining relative paths
                zip_file.write(file_path, os.path.relpath(file_path, folder_path))
                
def _sha256_to_uuid4(data, conversion_map, repo_path):
    for key, value in data.items():
        # Folder
        if key == 'k':
            if value is not None:
                already_converted = False
                if conversion_map.get(data[key], None) is not None:
                    new_key = conversion_map[data[key]]
                    already_converted = True
                else:
                    new_key = str(uuid.uuid4())
                
                if not already_converted:
                    CONVERSION_LOGGER.report('Updating key to existing uuid4 {} -> {}'.format(data[key], new_key))
                else:
                    CONVERSION_LOGGER.report('Converting key {} -> {}'.format(data[key], new_key))
                
                conversion_map[str(data[key])] = new_key
                
                old_key = str(data[key])
                data[key] = new_key
                
                if not already_converted:
                    file_path = os.path.join(repo_path, old_key)
                    mv_file_path = os.path.join(repo_path, new_key)
                    
                    if os.path.exists(file_path):
                        os.rename(file_path, mv_file_path)
                    else:
                        raise Exception('File does not exist at {}'.format(file_path))
        else:
            _sha256_to_uuid4(data[key], conversion_map, repo_path)
            
            
def _uuid4_to_sha256(data, conversion_map, repo_path):
    for key, value in data.items():
        # Folder
        if key == 'k':
            if value is not None:
                already_converted = False
                if conversion_map.get(data[key], None) is not None:
                    new_key = conversion_map[data[key]]
                    already_converted = True
                else:
                    # Read file and generate sha256 hash
                    file_path = os.path.join(repo_path, data[key])
                    with open(file_path, 'rb') as file:               
                        sha256_hash = hashlib.sha256()
                        while chunk := file.read(4096):
                            sha256_hash.update(chunk)
                        
                        new_key = sha256_hash.hexdigest()
                
                if not already_converted:
                    CONVERSION_LOGGER.report('Updating key to existing uuid4 {} -> {}'.format(data[key], new_key))
                else:
                    CONVERSION_LOGGER.report('Converting key {} -> {}'.format(data[key], new_key))
                
                conversion_map[str(data[key])] = new_key
                
                old_key = str(data[key])
                data[key] = new_key
                
                if not already_converted:
                    file_path = os.path.join(repo_path, old_key)
                    mv_file_path = os.path.join(repo_path, new_key)
                    
                    if os.path.exists(file_path):
                        os.rename(file_path, mv_file_path)
                    else:
                        raise Exception('File does not exist at {}'.format(file_path))
        else:
            _uuid4_to_sha256(data[key], conversion_map, repo_path)
    
    
def _convert_archive_export(archive_path : str, key_format: str, convert_function : callable) -> str:
    """
        Converts archive key format given a conversion function
        
        - Extract the archive
        - Update the metadata.json file
        - Generate new key_format for the files
        - Zip the files
        
        :param archive_path: path to the export archive
        :param key_format: new key format
        :param convert_function: conversion function
        
        :returns: converted archive path
        
    """
    
    if not os.path.exists(archive_path):
        raise Exception('Archive does not exist at {}'.format(archive_path))
    
    root_dir = os.path.dirname(archive_path)
    temp_data = os.path.join(root_dir, 'temp')
    
    if not os.path.exists(temp_data):
        os.mkdir(temp_data)
    else:
        raise Exception('Temporary directory already exists at {}'.format(temp_data))
    
    CONVERSION_LOGGER.report('Extracting archive to {}'.format(temp_data))
    # Extract the archive
    with zipfile.ZipFile(archive_path, 'r') as zip_ref:
        zip_ref.extractall(temp_data)
        
    # Update the key_format in the metadata.json file

    
    metadata_path = os.path.join(temp_data, 'metadata.json')
    if not os.path.exists(metadata_path):
        raise Exception('metadata.json file does not exist at {}'.format(metadata_path))
    
    db_path = os.path.join(temp_data, 'db.sqlite3')
    if not os.path.exists(db_path):
        raise Exception('db.sqlite3 file does not exist at {}'.format(db_path))
    
    repo_path = os.path.join(temp_data, 'repo')
    if not os.path.exists(repo_path):
        raise Exception('repo directory does not exist at {}'.format(repo_path))
    
    metadata = None
    
    CONVERSION_LOGGER.report('Updating metadata.json file')
    try:
        with open(metadata_path, 'r') as metadata_file:
            metadata = json.load(metadata_file)
            metadata['key_format'] = key_format
    except Exception as e:
        raise Exception('Could not read metadata.json file') from e
    
    if metadata is None:
        raise Exception('Could not read metadata.json file')
    
    try:
        with open(metadata_path, 'w') as metadata_file:
            json.dump(metadata, metadata_file)
    except Exception as e:
        raise Exception('Could not write to metadata.json file') from e
    
    conversion_map = {}
    new_metadata = {}
    
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM db_dbnode')
            row = cursor.fetchone()
            while row:
                repository_metadata = json.loads(row[10])
                if repository_metadata:
                    CONVERSION_LOGGER.report("Converting node: {}".format(row[0]))
                    convert_function(repository_metadata, conversion_map, repo_path)
                    new_metadata[row[0]] = json.dumps(repository_metadata)
                
                row = cursor.fetchone()
            
            CONVERSION_LOGGER.report('Updating databae')
            for key, value in new_metadata.items():
                CONVERSION_LOGGER.report('Updating node {}'.format(key))
                cursor.execute('UPDATE db_dbnode SET repository_metadata = ? WHERE id = ?', (value, key))
            
    except Exception as e:
        shutil.rmtree(temp_data)
        raise Exception('Can\'t perform conversion') from e
    
    old_file_name = os.path.basename(archive_path)
    new_archive_path = os.path.join(root_dir, f"{key_format}-{old_file_name}")
    
    CONVERSION_LOGGER.report(f'Generating new archive: {new_archive_path}')
    _zip_directory(temp_data, new_archive_path)
    CONVERSION_LOGGER.report('Cleaning up')
    shutil.rmtree(temp_data)
    
    
    CONVERSION_LOGGER.report('Conversion complete')
    return new_archive_path