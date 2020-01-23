from flask import render_template, request
from recall import app
from sqlalchemy import create_engine
from sqlalchemy_utils import database_exists, create_database
import pandas as pd
import psycopg2
from .a_model import ModelIt

user = 'katherinewood' #add your username here (same as previous postgreSQL)                      
host = 'localhost'
dbname = 'insight'
db = create_engine('postgres://%s%s/%s'%(user,host,dbname))
con = None
con = psycopg2.connect(database = dbname, user = user)

@app.route('/')
@app.route('/index')
def index():
    return render_template("index.html",
       title = 'Home', user = { 'nickname': 'Katherine' },
       )

@app.route('/db')
def recall_page():
    sql_query = """                                                                       
                SELECT * FROM recalls LIMIT 20;          
                """
    query_results = pd.read_sql_query(sql_query,con)
    recalls = ""
    for i in range(0,10):
        recalls += query_results.iloc[i]['description']
        recalls += "<br>"
    return recalls

@app.route('/recalls')
def recall_page_fancy():
    sql_query = """
               SELECT recallid, title, hazards_name, description FROM recalls LIMIT 5;
                """
    query_results=pd.read_sql_query(sql_query,con)
    recalls = []
    for i in range(query_results.shape[0]):
        #print(dict(recallid=query_results.iloc[i]['recallid'], title=query_results.iloc[i]['title'], hazards_name=query_results.iloc[i]['hazards_name'], description=query_results.iloc[i]['description']))
        recalls.append(dict(recallid=query_results.iloc[i]['recallid'], 
            title=query_results.iloc[i]['title'], 
            hazards_name=query_results.iloc[i]['hazards_name'], 
            description=query_results.iloc[i]['description']))
    return render_template('recalls.html', recalls=recalls)

@app.route('/input')
def recall_input():
    return render_template("input.html")

@app.route('/output')
def recall_output():
  #pull 'brand' from input field and store it
  brand = request.args.get('brand')
    #just select the Cesareans  from the birth dtabase for the month that the user inputs
  query = "SELECT recallid, title, hazards_name, description FROM recalls WHERE title LIKE '%" + brand + "%'"
  #print(query)
  query_results=pd.read_sql_query(query,con)
  #print(query_results)
  the_result = ''
  recalls = []
  for i in range(0,query_results.shape[0]):
        recalls.append(dict(recallid=query_results.iloc[i]['recallid'], 
                    title=query_results.iloc[i]['title'], 
                    hazards_name=query_results.iloc[i]['hazards_name'], 
                    description=query_results.iloc[i]['description']))
        the_result = ModelIt(brand,recalls)
  return render_template("model_output.html", recalls = recalls, the_result = the_result)

# @app.route('/output')
# def recall_output():
#   #pull 'brand' from input field and store it
#   brand = request.args.get('brand')
#     #just select the Cesareans  from the birth dtabase for the month that the user inputs
#   query = "SELECT recallid, title, hazards_name, description FROM recalls WHERE title LIKE '%" + brand + "%'"
#   #print(query)
#   query_results=pd.read_sql_query(query,con)
#   #print(query_results)
#   the_result = ''
#   recalls = []
#   for i in range(0,query_results.shape[0]):
#         recalls.append(dict(recallid=query_results.iloc[i]['recallid'], 
#                     title=query_results.iloc[i]['title'], 
#                     hazards_name=query_results.iloc[i]['hazards_name'], 
#                     description=query_results.iloc[i]['description']))
#   return render_template("output.html", recalls = recalls, the_result = the_result)
