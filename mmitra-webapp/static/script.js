'use strict';

var columnToIndexMap = {
    'filterChannelType': 3,
    'filterChannelName': 4,
    'filterIncomeBracket': 5,
    'filterCallSlot': 6,
    'filterEntryDate': 7,
    'filterEducation': 9,
};

function download_table_as_csv(table_id, separator = ',') {
    // Select rows from table_id
    var rows = document.querySelectorAll('table#' + table_id + ' tr');
    // Construct csv
    var csv = [];
    for (var i = 0; i < rows.length; i++) {
        if (rows[i].style.display == "none") {
          continue;
        }
        var row = [], cols = rows[i].querySelectorAll('td, th');
        for (var j = 0; j < cols.length; j++) {
            // Clean innertext to remove multiple spaces and jumpline (break csv)
            var data = cols[j].innerText.replace(/(\r\n|\n|\r)/gm, '').replace(/(\s\s)/gm, ' ')
            // Escape double-quote with double-double-quote (see https://stackoverflow.com/questions/17808511/properly-escape-a-double-quote-in-csv)
            data = data.replace(/"/g, '""');
            // Push escaped string
            row.push('"' + data + '"');
        }
        csv.push(row.join(separator));
    }
    var csv_string = csv.join('\n');
    // Download it
    var filename = 'export_' + table_id + '_' + new Date().toLocaleDateString() + '.csv';
    var link = document.createElement('a');
    link.style.display = 'none';
    link.setAttribute('target', '_blank');
    link.setAttribute('href', 'data:text/csv;charset=utf-8,' + encodeURIComponent(csv_string));
    link.setAttribute('download', filename);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
}


function filterTableButton() {
  // Declare variables
  var input, inputValue, filter, filters, table, tr, td, i, j, txtValue;
  filters = []
  for(const mapEntry in columnToIndexMap) {
    var inputId = `${mapEntry}`;
    input = document.getElementById(inputId);
    inputValue = input.value.toUpperCase();
    if (inputValue != "") {
      filters.push([inputId, inputValue]);
    }
  }
  console.log(filters);
  table = document.getElementById("beneficiariesTable");
  tr = table.getElementsByTagName("tr");

  // Loop through all table rows, and hide those who don't match the search query
  for (i = 1; i < tr.length; i++) {
    var shouldFilter = false;
    for (j = 0; j < filters.length; j++) {
      inputId = filters[j][0];
      filter = filters[j][1];
      console.log(inputId + " : " + filter);
      td = tr[i].getElementsByTagName("td")[columnToIndexMap[inputId]];
      if (td) {
        txtValue = td.textContent || td.innerText;
        if (txtValue.toUpperCase().indexOf(filter) <= -1) {
          shouldFilter = true;
        }
      }
    }
    if (shouldFilter) {
      tr[i].style.display = "none";
    } else {
      tr[i].style.display = "";
    }
  }
}

function retrievePredictions() {
	gapi.load('auth2,signin2', function() {
        var auth2 = gapi.auth2.init();
        auth2.then(function() {
          // Current values
          var isSignedIn = auth2.isSignedIn.get();
          var currentUser = auth2.currentUser.get();
          if (!isSignedIn) {
            window.location.href = "http://localhost:8080";
          } else {
          	var form = document.getElementById("retrievePredictionsForm");
      			var profile = currentUser.getBasicProfile();
      			console.log("submitting form with " + profile.getEmail());

      			var authId = document.createElement("input");
      			authId.setAttribute("name", "authId");
      			authId.setAttribute("value", profile.getEmail());
      			authId.setAttribute("type", "hidden");
      			form.appendChild(authId);
      			console.log(form);
      			form.submit();
          }
        });
    });
}

function isUserSignedIn() {
  document.getElementById("retrievePredictions").style.display = "none";
  gapi.load('auth2', function() {
      var auth2 = gapi.auth2.init();
      auth2.then(function() {
        // Current values
        var isSignedIn = auth2.isSignedIn.get();
        console.log('is signed in? ', isSignedIn);
        var currentUser = auth2.currentUser.get();
        if (isSignedIn == true) {
          onSignIn(currentUser);
        }
      });
  });
}

function onSignIn(googleUser) {
  console.log("User signed in.")

  // Useful data for your client-side scripts:
  var profile = googleUser.getBasicProfile();

  // The ID token you need to pass to your backend:
  var id_token = googleUser.getAuthResponse().id_token;

  document.getElementById("unauthorized").style.display = "none";		
  document.getElementById("login").style.display = "none";
  document.getElementById("retrievePredictions").style.display = "block";
  document.getElementById("signOut").style.display = "inline-block";
}

function signOut() {
  var auth2 = gapi.auth2.getAuthInstance();
  auth2.signOut().then(function () {
    console.log('User signed out.');
    location.reload();
  });
}