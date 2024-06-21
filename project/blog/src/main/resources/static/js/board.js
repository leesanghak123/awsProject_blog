let index = {
	init: function() {
		// btn-save를 찾아서 클릭이 일어나면 save를 호출
		$("#btn-save").on("click", () => { // function(){} 대신 ()=>{}를 사용하는 이유는 this를 바인딩하기 위해서 사용
			this.save();
		});
		$("#btn-delete").on("click", () => { // function(){} 대신 ()=>{}를 사용하는 이유는 this를 바인딩하기 위해서 사용
			this.deleteById();
		});
		$("#btn-update").on("click", () => { // function(){} 대신 ()=>{}를 사용하는 이유는 this를 바인딩하기 위해서 사용
			this.update();
		});
		$("#btn-reply-save").on("click", () => { // function(){} 대신 ()=>{}를 사용하는 이유는 this를 바인딩하기 위해서 사용
			this.replySave();
		});
		$("#btn-good").on("click", () => { // function(){} 대신 ()=>{}를 사용하는 이유는 this를 바인딩하기 위해서 사용
			this.good();
		});
        $(document).on("click", ".btn-toggle-reply-reply", function() {
            const replyId = $(this).data("reply-id");
            $(`#reply-reply-form-${replyId}`).toggle();
        });
        $(document).on("click", ".btn-reply-reply-save", function() {
            const replyId = $(this).data("reply-id");
            index.replyReplySave(replyId);
        });
        $("#test").on("click", () => { // function(){} 대신 ()=>{}를 사용하는 이유는 this를 바인딩하기 위해서 사용
			this.test();
		});
	},

	save: function() {
		if (!this.validateBoardForm()) {
            return;
        }
		//alert('user의 save함수 호출됨');
		let data = {
			title: $("#title").val(),
			content: $("#content").val()
		};

		//console.log(data);
		
		// ajax 호출 시 default가 비동기 호출
		// ajax 통신을 이용해서 3개의 데이터를 json으로 변경하여 insert 요청
		// ajax가 통신을 성공하고 서버가 json을 리턴해주면 자동으로 자바 오브젝트로 변환해준다 (밑의 dataType:"json" 를 안적어도 된다는 말)
		$.ajax({
			type:"POST", // 회원가입 할거니까 post(insert)
			url:"/api/board",
			data:JSON.stringify(data), // http body 데이터, 위의 data 자바 오브젝트를 JSON으로 던져줄 것
			contentType:"application/json; charset=utf-8", // 위의 코드와 세트, body데이터가 어떤 타입인지(MIME)
			dataType:"json" // 요청을 서버로해서 응답이 왔을 때 기본적으로 모든 것이 문자열 (생긴게 json이라면) => javascript 오브젝트로 변경
		}).done(function(resp){
			if(resp.status === 500) {	// globalException에서 에러가 나면 500에러로 던져 줌
				alert("글쓰기가 실패하였습니다.");
				location.reload();
			}else {
				alert("글쓰기가 완료되었습니다.");
				//console.log(resp)
				location.href="/";
			}
		}).fail(function(error){
			alert(JSON.stringify(error));
		});

	},
	
	deleteById: function() {
		let id =$("#id").val();
		
		$.ajax({
			type:"DELETE",
			url:"/api/board/"+id,
			dataType:"json" // 요청을 서버로해서 응답이 왔을 때 기본적으로 모든 것이 문자열 (생긴게 json이라면) => javascript 오브젝트로 변경
		}).done(function(resp){
			alert("삭제가 완료되었습니다.");
			location.href="/"; // /경로로 돌아옴
		}).fail(function(error){
			alert(JSON.stringify(error));
		});

	},
	
	update: function() {
		if (!this.validateBoardForm()) {
            return;
        }
		let id = $("#id").val();
		
		let data = {
			title: $("#title").val(),
			content: $("#content").val()
		};

		$.ajax({
			type:"PUT", // 수정 할거니까 PUT
			url:"/api/board/"+id,
			data:JSON.stringify(data), // http body 데이터, 위의 data 자바 오브젝트를 JSON으로 던져줄 것
			contentType:"application/json; charset=utf-8", // 위의 코드와 세트, body데이터가 어떤 타입인지(MIME)
			dataType:"json" // 요청을 서버로해서 응답이 왔을 때 기본적으로 모든 것이 문자열 (생긴게 json이라면) => javascript 오브젝트로 변경
		}).done(function(resp){
			if(resp.status === 500) {	// globalException에서 에러가 나면 500에러로 던져 줌
				alert("글쓰기가 실패하였습니다.");
				location.reload();
			}else {
				alert("글쓰기가 완료되었습니다.");
				//console.log(resp)
				location.href="/";
			}
		}).fail(function(error){
			alert(JSON.stringify(error));
		});

	},
	
	replySave: function() {
		if (!this.validateReplyForm()) {
            return;
        }

		let data = {
			// boardId: $("#boardId").val(), 어디 게시물의 댓글인지 알기 위함인데 밑에 url:'/api/board/${data.boardId}/reply',가 있어서 필요없음
			userId: $("#userId").val(),
			boardId: $("#boardId").val(),
			content: $("#reply-content").val() // val은 int형
		};
		
		$.ajax({
			type:"POST", 
			url:`/api/board/${data.boardId}/reply`,  // `(백틱)를 사용하는 이유는 게시물 번호를 동적으로 받기 위함 (작은 따옴표가 아니고 !옆에 있는거)
			data:JSON.stringify(data), 
			contentType:"application/json; charset=utf-8", 
			dataType:"json" 
		}).done(function(resp){
			if(resp.status === 500) {	// globalException에서 에러가 나면 500에러로 던져 줌
				alert("댓글작성이 실패하였습니다.");
				location.reload();
			}else {
				alert("댓글작성이 완료되었습니다.");
				//console.log(resp)
				location.href=`/board/${data.boardId}`;
			}
		}).fail(function(error){
			alert(JSON.stringify(error));
		});

	},
	
	replyDelete: function(boardId, replyId) {

		$.ajax({
			type:"DELETE",
			url:`/api/board/${boardId}/reply/${replyId}`,
			dataType:"json" 
		}).done(function(resp){
			alert("댓글삭제 성공");
			location.href=`/board/${boardId}`;
		}).fail(function(error){
			alert(JSON.stringify(error));
		});

	},
	
	good: function(boardId) {
        $.ajax({
            type: "POST",
            url: `/api/board/${boardId}/good`,
            dataType: "json"
        }).done(function(resp) {
            location.reload(); // 페이지를 새로고침하여 좋아요 수 업데이트
        }).fail(function(error) {
            alert(JSON.stringify(error));
        });
    },

    replyReplySave: function(replyId) {
        let data = {
            userId: $("#userId").val(),
            replyId: replyId,
            content: $(`#reply-reply-content-${replyId}`).val()
        };

        $.ajax({
            type: "POST",
            url: `/api/reply/${replyId}/reply`,
            data: JSON.stringify(data),
            contentType: "application/json; charset=utf-8",
            dataType: "json"
        }).done(function(resp) {
            location.reload();
        }).fail(function(error) {
            alert(JSON.stringify(error));
        });
    },
    
    replyReplyDelete: function(boardId, replyId, replyReplyId) {

		$.ajax({
			type:"DELETE",
			url:`/api/board/${boardId}/reply/${replyId}/${replyReplyId}`,
			dataType:"json" 
		}).done(function(resp){
			alert("댓글삭제 성공");
			location.href=`/board/${boardId}`;
		}).fail(function(error){
			alert(JSON.stringify(error));
		});

	},
	
	// 유효성 검사 -------------------------------------------------------------------------------------------------------------

		validateBoardForm: function() {
	        let title = $("#title").val();
	        let content = $("#content").val();

	        if (title === "" || content === "") {
	            alert("모두 작성해주세요.");
	            return false;
	        }

	        return true;
	    },
		
		validateReplyForm: function() {
	        let replyContent = $("#reply-content").val();

	        if (replyContent === "") {
	            alert("댓글을 작성해주세요.");
	            return false;
	        }

	        return true;
	    },
		
}

index.init();