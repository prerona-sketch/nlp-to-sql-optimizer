async function generateSQL() {

    let query = document.getElementById("userInput").value;

    document.getElementById("sqlOutput").innerText = "Generating...";
    document.getElementById("tableOutput").innerHTML = "";

    try {

        let response = await fetch("http://127.0.0.1:8000/query", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({ question: query })
        });

        let data = await response.json();

        document.getElementById("sqlOutput").innerText = data.sql;

        let tableHTML = "<table>";

        // Headers
        let keys = Object.keys(data.result[0]);
        tableHTML += "<tr>";
        keys.forEach(key => {
            tableHTML += `<th>${key}</th>`;
        });
        tableHTML += "</tr>";

        // Rows
        data.result.forEach(row => {
            tableHTML += "<tr>";
            keys.forEach(key => {
                tableHTML += `<td>${row[key]}</td>`;
            });
            tableHTML += "</tr>";
        });

        tableHTML += "</table>";

        document.getElementById("tableOutput").innerHTML = tableHTML;

    } catch (error) {
        document.getElementById("sqlOutput").innerText = "Error connecting to server";
    }
}

function clearAll() {
    document.getElementById("userInput").value = "";
    document.getElementById("sqlOutput").innerText = "";
    document.getElementById("tableOutput").innerHTML = "";
}