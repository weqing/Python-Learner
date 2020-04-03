# -*- coding: utf-8 -*-

from sqlalchemy.orm import sessionmaker
from models import WeiBoTable, db_connect, create_weibo_table


# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: http://doc.scrapy.org/en/latest/topics/item-pipeline.html


class WeiboPipeline(object):
    def __init__(self):
        engine = db_connect()
        create_weibo_table(engine)
        self.Session = sessionmaker(bind=engine)

    def process_item(self, item, spider):
        session = self.Session()

        weibo = WeiBoTable(**item)
        try:
            if self.post_exist(session, WeiBoTable, item):
                session.close()
                return item

            session.add(weibo)
            session.commit()
        except Exception as e:
            print "pipelines", e
            session.rollback()
            raise
        finally:
            session.close()

        return item

    def post_exist(self, session, Sobj, item):
        content_id = session.query(Sobj).filter_by(content_id=item['content_id']).first()

        if content_id:
            return True
        return False


