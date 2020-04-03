#!/usr/bin/env python
#-*-coding:utf-8-*-
from sqlalchemy import create_engine, Column, Integer, String, Text
from sqlalchemy.engine.url import URL
from sqlalchemy.ext.declarative import declarative_base

import settings

DeclarativeBase = declarative_base()

def db_connect():
    return create_engine(URL(**settings.DATABASE), connect_args={'charset': 'utf8'})

def create_weibo_table(engine):
    DeclarativeBase.metadata.create_all(engine)

class WeiBoTable(DeclarativeBase):
    __tablename__ = "weibo"
    id = Column(Integer, primary_key=True)
    keyword = Column(String(512))
    user_id = Column(String(64))
    user_name = Column(String(512))
    content_id = Column(String(64), unique=True)
    content = Column(Text)
    repost_num = Column(Integer)
    ptime = Column(String(128))
    scrapy_time = Column(Integer)
