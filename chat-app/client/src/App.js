import React, { useEffect, useState } from 'react';
import { MessageBox } from 'react-chat-elements'
import 'react-chat-elements/dist/main.css';
import './App.css'
import parse from 'html-react-parser';
import DarkModeToggle from "react-dark-mode-toggle";


function App() {

	const [messages, setMessages] = useState([
		{
			title: "Bot",
			text: "Hi! I'm XAI-BOT. What is your name?",
			date: new Date()
		}
	])

	const [endOfConversation, setEndOfConversation] = useState(false)
	const [disable, setDisable] = useState(false)
	const [userMessage, setUserMessage] = useState("")
	const [darkMode, setDarkMode] = useState(false)

	useEffect(() => {
		initializeTheme()
	}, [])

	const initializeTheme = async () => {
		if (darkMode) {
			var botMesssages = document.getElementsByClassName("rce-mbox");
			console.log(botMesssages)
			for (let i = 0; i < botMesssages.length; i++) {
				botMesssages[i].className += " darkModeBotMessage";
			}
		} else {
			var botMesssages = document.getElementsByClassName("rce-mbox");
			for (let i = 0; i < botMesssages.length; i++) {
				if (botMesssages[i].className.includes("rce-mbox-right")) {
					console.log("User message")
				} else {
					botMesssages[i].className = "rce-mbox";
				}
			}
		}
	}

	const stickToTheme = async () => {
		if (darkMode) {
			var botMesssages = document.getElementsByClassName("rce-mbox");
			console.log(botMesssages)
			for (let i = 0; i < botMesssages.length; i++) {
				if (botMesssages[i].className.includes("rce-mbox-right")) {
					console.log("User message")
					botMesssages[i].className += " darkModeUserMessage";
				} else {
					botMesssages[i].className += " darkModeBotMessage";
				}
			}
		} else {
			var botMesssages = document.getElementsByClassName("rce-mbox");
			for (let i = 0; i < botMesssages.length; i++) {
				if (botMesssages[i].className.includes("rce-mbox-right")) {
					console.log("User message")
					botMesssages[i].className = "rce-mbox rce-mbox-right";
				} else {
					botMesssages[i].className = "rce-mbox";
				}
			}
		}
	}

	const stickToThemeForce = async (dark) => {
		if (dark) {
			var botMesssages = document.getElementsByClassName("rce-mbox");
			console.log(botMesssages)
			for (let i = 0; i < botMesssages.length; i++) {
				if (botMesssages[i].className.includes("rce-mbox-right")) {
					console.log("User message")
					botMesssages[i].className += " darkModeUserMessage";
				} else {
					botMesssages[i].className += " darkModeBotMessage";
				}
			}
		} else {
			var botMesssages = document.getElementsByClassName("rce-mbox");
			for (let i = 0; i < botMesssages.length; i++) {
				if (botMesssages[i].className.includes("rce-mbox-right")) {
					console.log("User message")
					botMesssages[i].className = "rce-mbox rce-mbox-right";
				} else {
					botMesssages[i].className = "rce-mbox";
				}
			}
		}
	}

	const scrollToBottom = () => {
		// var myDiv = document.getElementById("message-container");
		// myDiv.scrollTop = myDiv.lastChild.offsetTop
		const scrollContainer = document.getElementById('message-container');
		console.log(scrollContainer.scrollHeight)
		scrollContainer.scrollTo({
			top: scrollContainer.scrollHeight * 500,
			left: 0,
			behavior: 'smooth'
		});
	}

	const sendMessage = async () => {
		console.log("Sending message...")
		console.log(userMessage)
		setMessages(oldArray => [...oldArray, {
			title: "You",
			text: userMessage,
			date: new Date()
		},]);
		setUserMessage("")
		setMessages(oldArray => [...oldArray, {
			title: "Bot",
			intention: "wants_plot",
			
			url: "/loading-dots_4.gif",
			date: new Date()
		},]);
		let response = await fetch('http://localhost:5000/api/get-message', {
			method: "POST",
			headers: {
				"Content-Type": "application/json"
			},
			body: JSON.stringify({
				"text": userMessage
			})
		})

		response = await response.json();
		console.log(response.data)


		function replaceURLWithHTMLLinks(text) {
			var exp = /(\b(https?|ftp|file):\/\/[-A-Z0-9+&@#\/%?=~_|!:,.;]*[-A-Z0-9+&@#\/%=~_|])/ig;
			text = text.replaceAll('\n', "<br/>");
			return text.replace(exp, "<a target='_blank' href='$1'>$1</a>");
		}
		try {
			response.data.msg = replaceURLWithHTMLLinks(response.data.msg)
			response.data.txt = parse("<div style{'margin-left:-10px;'}>" + response.data.msg + "</div>");
			response.data.modified = true;
			console.log(response.data.msg)
		} catch (e) {
			console.log("ignoring")
		}


		// 
		setMessages(oldArray => {
			console.log("The array is:")
			
			console.log(JSON.stringify(oldArray))
			let tmp = [];
			for (var i = 0; i < oldArray.length - 1; i++){
				tmp.push(oldArray[i]);
			}
			oldArray = tmp;
			return [...oldArray, {
				title: "Bot",
				text: response.data.msg,
				date: new Date(),
				intention: response.data.intention,
				options: response.data.options,
				end_of_converstaion: response.data.end_of_converstaion,
				url: response.data.url || false,
				modified: response.data.modified,
				txt: response.data.txt
			},]
		});

		setEndOfConversation(response.data.end_of_converstaion)
		stickToTheme();
		setTimeout(() => {
			scrollToBottom()
		}, 100)
	}

	const handleSubmit = async (e) => {
		e.preventDefault();
		await sendMessage();
	};

	const handleKeyPress = async (e) => {
		if (e.key === 'Enter') {
			e.preventDefault();
			await sendMessage();
		}
	}

	const handleOptionSelect = async (option) => {
		console.log("Sending option message...")

		setMessages(oldArray => [...oldArray, {
			title: "You",
			text: option,
			date: new Date()
		},]);

		setMessages(oldArray => [...oldArray, {
			title: "Bot",
			intention: "wants_plot",
			
			url: "/loading-dots_4.gif",
			date: new Date()
		},]);

		let response = await fetch('http://localhost:5000/api/get-message', {
			method: "POST",
			headers: {
				"Content-Type": "application/json"
			},
			body: JSON.stringify({
				"text": option
			})
		})

		response = await response.json();
		console.log(response.data)

		function replaceURLWithHTMLLinks(text) {
			var exp = /(\b(https?|ftp|file):\/\/[-A-Z0-9+&@#\/%?=~_|!:,.;]*[-A-Z0-9+&@#\/%=~_|])/ig;
			text = text.replaceAll('\n', "<br/>");
			return text.replace(exp, "<a target='_blank' href='$1'>$1</a>");
		}

		try {
			response.data.msg = replaceURLWithHTMLLinks(response.data.msg)
			response.data.txt = parse("<div style{'margin-left:-10px;'}>" + response.data.msg + "</div>");
			response.data.modified = true;
			console.log(response.data.msg)
		} catch (e) {
			console.log("ignoring")
		}


		setMessages(oldArray => {
			console.log("The array is:")
			
			console.log(JSON.stringify(oldArray))
			let tmp = [];
			for (var i = 0; i < oldArray.length - 1; i++){
				tmp.push(oldArray[i]);
			}
			oldArray = tmp;
			return [...oldArray, {
			title: "Bot",
			text: response.data.msg,
			date: new Date(),
			intention: response.data.intention,
			options: response.data.options,
			end_of_converstaion: response.data.end_of_converstaion,
			url: response.data.url || false,
			modified: response.data.modified,
			txt: response.data.txt
		},]
	});

		setEndOfConversation(response.data.end_of_converstaion)
		stickToTheme();

		setTimeout(() => {
			scrollToBottom()
		}, 100)
	}

	const handleEndOfConversation = async (option) => {
		setDisable(true);
		console.log("Converstation has ended")
		setMessages(oldArray => [...oldArray, {
			title: "You",
			text: option,
			date: new Date()
		},]);
		setMessages(oldArray => [...oldArray, {
			title: "Bot",
			text: "Thank you for your feedback!",
			date: new Date()
		},]);
		let response = await fetch('http://localhost:5000/api/save-conversation', {
			method: "POST",
			headers: {
				"Content-Type": "application/json"
			},
			body: JSON.stringify({
				"feedback": option,
				"conversation": messages
			})
		})

		stickToTheme();

		setTimeout(() => {
			scrollToBottom()
		}, 100)
	}

	return (
		<div className={darkMode ? "App dark-app" : "App"}>
			<div className="toggle-container">
			<p id="switch-mode-text">Switch mode</p>
				<label class="switch">
					<input type="checkbox" onChange={() => { setDarkMode(!darkMode); stickToThemeForce(!darkMode) }} />
					<span class="slider round"></span>
				</label>
			</div>
			<div className={darkMode ? "chat-container dark-chat-container" : "chat-container"}>

				<div className="messages-container" id="message-container">
					<div className={darkMode ? "header dark-header" : "header"}>
						XAI - BOT
					</div>
					{
						messages.map((msg) => {
							return (
								<>
									{
										msg.end_of_converstaion
											?
											null
											:
											<MessageBox
												avatar={msg.title === 'Bot' ? "/bot_avatar.png" : false}
												title={msg.title}
												titleColor={msg.title === 'Bot' ? 'red' : 'black'}
												position={msg.title === 'Bot' ? 'left' : 'right'}
												text={msg.modified ? msg.txt : msg.text}
												type={msg.intention === 'wants_plot' ? 'photo' : 'text'}
												data={
													msg.intention === 'wants_plot' ? {
														uri: msg.url
													} : false}
											/>
									}

									{
										msg.intention === 'wants_list' || msg.end_of_converstaion
											?
											<div className="option-container">
												{
													msg.options.map((option) => {
														return (
															<button key={option} disabled={disable} onClick={() => { msg.end_of_converstaion ? handleEndOfConversation(option) : handleOptionSelect(option) }}>
																{option}
															</button>
														)
													})
												}
											</div>
											:
											null
									}

								</>
							)
						})
					}
				</div>

				<div className={darkMode ? "input-container dark-input-container" : "input-container"}>
					{
						endOfConversation
							?
							<p> This conversation has ended </p>
							:
							<>
								<input
									placeholder="Type here..."
									value={userMessage}
									onChange={(e) => setUserMessage(e.target.value)}
									onKeyPress={(e) => handleKeyPress(e)}
								/>
								<button
									disabled={false}
									onClick={(e) => handleSubmit(e)}
								> Send </button>

							</>
					}

				</div>

			</div>

		</div>
	);
}

export default App;
