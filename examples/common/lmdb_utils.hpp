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


#pragma once
#include <lmdb.h>
#include <stdint.h>
#include <string>
#include <vector>

namespace lmdb_utils {

    enum class Mode { 
        READ,
        WRITE,
        NEW
    };

    class Cursor {
    public:
        Cursor() { }
        virtual ~Cursor() { }
        virtual void SeekToFirst() = 0;
        virtual void Next() = 0;
        virtual std::string key() = 0;
        virtual std::string value() = 0;
        virtual bool valid() = 0;
    };

    class Transaction {
    public:
        Transaction() { }
        virtual ~Transaction() { }
        virtual void Put(const std::string& key, const std::string& value) = 0;
        virtual void Commit() = 0;
    };

    class DB {
    public:
        DB() { }
        virtual ~DB() { }
        virtual void Open(const std::string& source, Mode mode) = 0;
        virtual void Close() = 0;
        virtual Cursor* NewCursor() = 0;
        virtual Transaction* NewTransaction() = 0;
    };

    class LMDBCursor : public Cursor {
    public:
        explicit LMDBCursor(MDB_txn* mdb_txn, MDB_cursor* mdb_cursor)
            : mdb_txn_(mdb_txn), mdb_cursor_(mdb_cursor), valid_(false) {
            SeekToFirst();
        }

        ~LMDBCursor() {
            mdb_cursor_close(mdb_cursor_);
            mdb_txn_abort(mdb_txn_);
        }

        void SeekToFirst() { Seek(MDB_FIRST); }
        void Next() { Seek(MDB_NEXT); }
        std::string key() {
            return std::string(static_cast<const char*>(mdb_key_.mv_data), mdb_key_.mv_size);
        }
        std::string value() {
            return std::string(static_cast<const char*>(mdb_value_.mv_data),
                mdb_value_.mv_size);
        }
        bool valid() { return valid_; }

    private:
        void Seek(MDB_cursor_op op) {
            int mdb_status = mdb_cursor_get(mdb_cursor_, &mdb_key_, &mdb_value_, op);
            if (mdb_status == MDB_NOTFOUND) {
                valid_ = false;
            }
            else if (mdb_status == MDB_SUCCESS) {
                valid_ = true;
            }
            else
                throw std::runtime_error("LMDB: Seek: mdb_cursor_get failed.");
        }

        MDB_txn* mdb_txn_;
        MDB_cursor* mdb_cursor_;
        MDB_val mdb_key_, mdb_value_;
        bool valid_;
    };

    class LMDBTransaction : public Transaction {
    public:
        explicit LMDBTransaction(MDB_env* mdb_env)
            : mdb_env_(mdb_env) { }
        void Put(const std::string& key, const std::string& value);
        void Commit();

    private:
        MDB_env* mdb_env_;
        std::vector<std::string> keys, values;

        void DoubleMapSize();
    };

    class LMDB : public DB {
    public:
        LMDB() : mdb_env_(NULL) { }
        ~LMDB() { Close(); }
        void Open(const std::string& source, Mode mode);
        void Close() {
            if (mdb_env_ != NULL) {
                mdb_dbi_close(mdb_env_, mdb_dbi_);
                mdb_env_close(mdb_env_);
                mdb_env_ = NULL;
            }
        }
        LMDBCursor* NewCursor();
        LMDBTransaction* NewTransaction();

    private:
        MDB_env* mdb_env_;
        MDB_dbi mdb_dbi_;
    };
}