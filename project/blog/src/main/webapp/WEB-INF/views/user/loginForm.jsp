<%@ page language="java" contentType="text/html; charset=UTF-8" pageEncoding="UTF-8"%>

<%@ include file="../layout/header.jsp"%>

<div class="container">

	<form action="/auth/loginProc" method="post" onsubmit="return loginValidation();">
		<div class="form-group">
			<label for="username">아이디</label> <input type="text" name="username" class="form-control" placeholder="Enter username" id="username">
		</div>

		<div class="form-group">
			<label for="password">비밀번호</label> <input type="password" name="password" class="form-control" placeholder="Enter password" id="password">
		</div>

		<div class="text-right">
			<button id="btn-login" class="btn btn-dark">로그인</button>
			<a href="https://kauth.kakao.com/oauth/authorize?client_id=6ece1eddc88f259fef887cb9732ed10a&redirect_uri=http://localhost:8000/auth/kakao/callback&response_type=code"><img height="38px"
				src="/image/kakao_login_button.png" /></a>
		</div>
	</form>
</div>

<%@ include file="../layout/footer.jsp"%>