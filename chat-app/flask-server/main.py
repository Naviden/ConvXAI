#Module imports
from flask import Flask, send_from_directory
from flask import render_template, jsonify, request
from flask_cors import CORS
import uuid 
from datetime import datetime
import csv
from dialogue_tracker import main, chatbot


# Creates a Flask application, named app
app = Flask(__name__,  static_url_path='')
cors = CORS(app)


def make_response(user_input):
	# resp = {
	# 	"intention":"wants_plot",
	# 	"url" : "/lime_plots/feat_67.png",
	# 	# "url" : "https://res-3.cloudinary.com/crunchbase-production/image/upload/c_lpad,h_256,w_256,f_auto,q_auto:eco/v1459804290/mkxozts4fsvkj73azuls.png",
	# 	"end_of_converstaion" : False,
	# }
	resp = chatbot(user_input)
	print(f'resp --> {resp}')

	return resp
	
# resp = chatbot(user_input)
# print(f'the stupid  final resp is : {resp}')
# return resp


def make_response_old(user_input):
	#intention = detect(user_input)
	intention = 'wants_answer'
	if intention == 'wants_plot':
		return {
			"intention":"wants_plot",
			"url" : "https://res-3.cloudinary.com/crunchbase-production/image/upload/c_lpad,h_256,w_256,f_auto,q_auto:eco/v1459804290/mkxozts4fsvkj73azuls.png",
			"end_of_converstaion" : False,
		}
	elif intention == 'wants_list':
		return {
			"msg":"here is the list:",
			"options" : ['AAAAAA','BBBBBB','CCCCCC', 'DDDDDDDD'],
			"intention" : "wants_list",
			"end_of_converstaion" : False,
		}
	elif intention == 'wants_answer':
		return {
			"msg" : 'here is your answer https://www.facebook.com \n Pizza \n Burger',
			"intention" : "wants_answer",
			"end_of_converstaion" : False,
		}
	elif intention == 'end_of_converstaion':
		return {
			"end_of_converstaion" : True,
			"options" : ['Like it','Hate it'],
		}

# Route to fetch frontend
@app.route('/')
def index():
    return jsonify({"msg": "BOT service test successfull!"})

# Route to get message
@app.route('/api/get-message', methods=['POST'])
def api_all():
	print("Fetching message")
	text = request.json['text']
	print("User message is: " + text)
	response = make_response(text)
	print(response)
	return jsonify({"data": response})

# Route to save conversation with feedback
@app.route('/api/save-conversation', methods=['POST'])
def save_conversation():

	conversation_id = uuid.uuid1()
	print(str(conversation_id))

	feedback = request.json['feedback']
	conversation = request.json['conversation']
	print(feedback)
	now = datetime.now()
	# ddmmYY H:M:S
	dt_string = now.strftime("%d-%m-%Y %H:%M:%S")
	print(dt_string)
	f = open("conversations/"+str(conversation_id)+' '+dt_string+".txt", "w")
	for msg in conversation:
		try:
			convo_line = msg["title"]+ ": "+msg["text"]
			print(convo_line)
			f.write(convo_line + '\n')
		except Exception as e:
			print(e)

	print("Writing to csv file")
	try:

		with open('document.csv','a') as csvfile:
			fieldnames = ['id','feedback']
			writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
			writer.writerow({'id':str(conversation_id)+' '+dt_string, 'feedback':feedback})
	except:
		with open('document.csv','w') as csvfile:
			fieldnames = ['id','feedback']
			writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
			writer.writerow({'id':str(conversation_id)+' '+dt_string, 'feedback':feedback})

	return jsonify({"data": "ok"})



# Run the application
if __name__ == "__main__":
	app.run(debug=True)