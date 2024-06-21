<%@ page language="java" contentType="text/html; charset=UTF-8" pageEncoding="UTF-8"%>

<%@ include file="../layout/header.jsp"%>

<div class="container">

	<form>
		<!-- username은 안고칠건데 username을 알아야지 회원정보 수정이 가능하므로 id값을 hidden으로 던져주기 위함(user.js) -->
		<input type="hidden" id="id" value="${principal.user.id}" />
		<div class="form-group">
			<label for="username">Username</label> <input type="text" value="${principal.user.username}" class="form-control" placeholder="Enter username" id="username" readonly>
		</div>

		<!--  oauth(카카오 로그인)이 없으면 패스워드 수정 가능 -->
		<c:if test="${empty principal.user.oauth}">
			<div class="form-group">
				<label for="password">Password</label> <input type="password" class="form-control" placeholder="Enter password" id="password">
			</div>
		</c:if>

		<div class="form-group">
			<label for="email">Email</label> <input type="email" value="${principal.user.email}" class="form-control" placeholder="Enter email" id="email" readonly>
		</div>

	</form>

	<div class="text-right">
		<button type="button" id="btn-update" class="btn btn-dark">저장</button>
	</div>
</div>

<script src="/js/user.js"></script>

<%@ include file="../layout/footer.jsp"%>