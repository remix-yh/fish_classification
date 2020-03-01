$(function(){	
	$('#myfile').change(function(e){

		var file = e.target.files[0];
		var reader = new FileReader();

		var cvs = document.getElementById('cvs1');
		var ctx = cvs.getContext('2d');

		if(file.type.indexOf("image") < 0){
			return false;
		}

		reader.onload = (function(file){
			return function(e){
				var img = new Image();
				img.src = e.target.result;
				img.onload = function() {
					ctx.drawImage(img, 0, 0, 224, 224);
				}
			};
		})(file);
		reader.readAsDataURL(file);
	});
});
function callPostMethod() {
		var data = {};
		var cvs = document.getElementById('cvs1');

		$.ajax(
			{ 
				url:"/api/fish-classification",
				type: "POST",
				data:JSON.stringify(cvs.toDataURL('image/jpeg').split('base64,')[1]),
				dataType:'json',
				contentType:'application/json'
			})
		.then(function (data) {
			$('#output').find("tr:gt(0)").remove();
			$('#output').append('<tr><td>' + data["result"]["0"]["name"] + '</td><td>' + data["result"]["0"]["prediction"] + '</td></tr>');
			$('#output').append('<tr><td>' + data["result"]["1"]["name"] + '</td><td>' + data["result"]["1"]["prediction"] + '</td></tr>');
			$('#output').append('<tr><td>' + data["result"]["2"]["name"] + '</td><td>' + data["result"]["2"]["prediction"] + '</td></tr>');
		}, function (e) {
				alert("error: " + e);
		});
}