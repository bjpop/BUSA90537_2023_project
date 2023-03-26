# This is an example of Python code that has a
# string containing HTML. You will need to be careful
# to make sure the HTML elements are suitably encoded.

DOC_TEMPLATE = """<!doctype html>
<html lang=en>
    <head>
        <meta charset=utf-8>
        <title>Pretty Python</title>
        <style>
            pre {{font-size: 16pt}}
            .variable {{color: black}}
            .comment {{color: green}}
            .keyword {{color: blue}}
            .string {{color: orange}}
            .number {{color: red}}
            .operator {{color: purple}}
        </style>
    </head>
    <body>
        <h1>Python code inspector</h1>
        <ul>
            <li><a href="#stats">Statistics</a></li>
            <li><a href="#code">Code</a></li>
        </ul>
           <div id="stats">
               <h2>Statistics</h2>
               {stats}
           </div>
           <div id="code">
               <h2>Python code</h2>
               {code}
           </div>
    </body>
</html>
"""
