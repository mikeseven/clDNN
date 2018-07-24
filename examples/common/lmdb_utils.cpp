/*
// Copyright (c) 2018 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
*/

#include "lmdb_utils.hpp"
#include <string>

using namespace lmdb_utils;

void LMDB::Open(const std::string& source, Mode mode)
{
    if (mdb_env_create(&mdb_env_) != MDB_SUCCESS)
        throw std::runtime_error("LMDB: mdb_env_create failed.");

    int flags = 0;
    if (mode == Mode::READ) {
      flags = MDB_RDONLY | MDB_NOTLS | MDB_NOSUBDIR;
    }

    if (mdb_env_open(mdb_env_, source.c_str(), flags, 0664) != MDB_SUCCESS)
        throw std::runtime_error("LMDB: Open: mdb_env_open failed.");
}

LMDBCursor* LMDB::NewCursor()
{
    MDB_txn* mdb_txn;
    MDB_cursor* mdb_cursor;
    if (mdb_txn_begin(mdb_env_, NULL, MDB_RDONLY, &mdb_txn) != MDB_SUCCESS)
        throw std::runtime_error("LMDB: NewCursor: mdb_txn_begin failed.");
    if (mdb_dbi_open(mdb_txn, NULL, 0, &mdb_dbi_) != MDB_SUCCESS)
        throw std::runtime_error("LMDB: NewCursor: mdb_dbi_open failed.");
    if (mdb_cursor_open(mdb_txn, mdb_dbi_, &mdb_cursor) != MDB_SUCCESS)
        throw std::runtime_error("LMDB: NewCursor: mdb_cursor_open failed.");
    return new LMDBCursor(mdb_txn, mdb_cursor);
}

LMDBTransaction* LMDB::NewTransaction() {
  return new LMDBTransaction(mdb_env_);
}

void LMDBTransaction::Put(const std::string& key, const std::string& value) {
  keys.push_back(key);
  values.push_back(value);
}

void LMDBTransaction::Commit() {
  MDB_dbi mdb_dbi;
  MDB_val mdb_key, mdb_data;
  MDB_txn *mdb_txn;

  // Initialize MDB variables
  if (mdb_txn_begin(mdb_env_, NULL, 0, &mdb_txn) != MDB_SUCCESS)
      throw std::runtime_error("LMDB: Commit: mdb_txn_begin failed.");
  if (mdb_dbi_open(mdb_txn, NULL, 0, &mdb_dbi) != MDB_SUCCESS)
      throw std::runtime_error("LMDB: Commit: mdb_dbi_open failed.");

  for (int i = 0; i < keys.size(); i++) {
    mdb_key.mv_size = keys[i].size();
    mdb_key.mv_data = const_cast<char*>(keys[i].data());
    mdb_data.mv_size = values[i].size();
    mdb_data.mv_data = const_cast<char*>(values[i].data());

    // Add data to the transaction
    int put_rc = mdb_put(mdb_txn, mdb_dbi, &mdb_key, &mdb_data, 0);
    if (put_rc == MDB_MAP_FULL) {
      // Out of memory - double the map size and retry
      mdb_txn_abort(mdb_txn);
      mdb_dbi_close(mdb_env_, mdb_dbi);
      DoubleMapSize();
      Commit();
      return;
    }
    // May have failed for some other reason
    if (put_rc != MDB_SUCCESS)
        throw std::runtime_error("LMDB: Commit: put_rc failed.");
  }

  // Commit the transaction
  int commit_rc = mdb_txn_commit(mdb_txn);
  if (commit_rc == MDB_MAP_FULL) {
    // Out of memory - double the map size and retry
    mdb_dbi_close(mdb_env_, mdb_dbi);
    DoubleMapSize();
    Commit();
    return;
  }
  // May have failed for some other reason
  if (commit_rc != MDB_SUCCESS)
      throw std::runtime_error("LMDB: Commit: commit_rc failed.");

  // Cleanup after successful commit
  mdb_dbi_close(mdb_env_, mdb_dbi);
  keys.clear();
  values.clear();
}

void LMDBTransaction::DoubleMapSize() {
  struct MDB_envinfo current_info;
  if (mdb_env_info(mdb_env_, &current_info) != MDB_SUCCESS)
      throw std::runtime_error("LMDB: DoubleMapSize: mdb_env_info failed.");
  size_t new_size = current_info.me_mapsize * 2;
  if (mdb_env_set_mapsize(mdb_env_, new_size) != MDB_SUCCESS)
      throw std::runtime_error("LMDB: DoubleMapSize: mdb_env_set_mapsize failed.");
}
