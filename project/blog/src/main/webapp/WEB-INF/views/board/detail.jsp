<%@ page language="java" contentType="text/html; charset=UTF-8" pageEncoding="UTF-8"%>

<%@ include file="../layout/header.jsp"%>

<div class="container">

	<div class="text-right">
		<button class="btn btn-dark" onclick="history.back()">돌아가기</button>
		<c:if test="${board.user.id==principal.user.id}">
			<a href="/board/${board.id}/updateForm" class="btn btn-warning">수정</a>
			<button id="btn-delete" class="btn btn-warning">삭제</button>
		</c:if>
	</div>
	<br /> <br />
	<div class="d-flex justify-content-between">
		<div>
			<p style="font-size: 20px;">
				<strong>${board.title}</strong>
			</p>
		</div>
		<div class="text-right badge">
			<span style="display: block;"> <input type="hidden" id="id" value="${board.id}"> 작성자: <i>${board.user.username}</i></span>
			<span style="display: block;"> <i><fmt:formatDate value="${board.createDate}" pattern="yyyy-MM-dd hh:mm:ss" /></i></span> 
			좋아요: <i>${board.good}</i></span> <span style="display: block;">
		</div>
	</div>
	<hr>
	<div>
		<div>${board.content}</div>
	</div>
	<div class="text-right">
		<c:choose>
			<c:when test="${isGoodPressed}">
				<button type="button" id="btn-good" class="btn btn-danger" onclick="index.good(${board.id})">좋아요♡</button>
			</c:when>
			<c:otherwise>
				<button type="button" id="btn-good" class="btn btn-outline-danger" onclick="index.good(${board.id})">좋아요♡</button>
			</c:otherwise>
		</c:choose>
	</div>
	<hr>
	<div class="card">
		<form>
			<input type="hidden" id="userId" value="${principal.user.id}" /> <input type="hidden" id="boardId" value="${board.id}" />
			<div class="card-body">
				<textarea id="reply-content" class="form-control" rows="1"></textarea>
			</div>
			<!-- form-control = 행 끝까지 채우기 -->
			<div class="card-footer text-right">
				<button type="button" id="btn-reply-save" class="btn btn-dark">등록</button>
				<!-- type="button" : from안에 있으면 기본적으로 submit으로 바뀌기 때문 -->
			</div>
		</form>
	</div>

	<br />

	<div class="card">
		<div class="card-header">comments</div>
		<ul id="reply-box" class="list-group">
			<c:forEach var="reply" items="${board.replys}">
				<li id="reply-${reply.id}" class="list-group-item d-flex justify-content-between">
					<div>${reply.content}</div>
					<div class="d-flex align-items-start">
						<div class="font-italic badge">작성자: ${reply.user.username} &nbsp;</div>
						<button type="button" class="btn-toggle-reply-reply badge" data-reply-id="${reply.id}">댓글쓰기</button>
						<c:if test="${reply.user.id == principal.user.id}">
							<button onClick="index.replyDelete(${board.id}, ${reply.id})" class="badge">삭제</button>
						</c:if>
					</div>
				</li>
				<div id="reply-reply-form-${reply.id}" class="card mt-2" style="display: none;">
					<div class="card-body">
						<textarea id="reply-reply-content-${reply.id}" class="form-control" rows="1"></textarea>
						<div class="d-flex justify-content-end mt-2">
							<button type="button" class="btn-reply-reply-save badge" data-reply-id="${reply.id}">대댓글 등록</button>
						</div>
					</div>
				</div>
				<c:forEach var="replyReply" items="${reply.replyReplies}">
					<li class="list-group-item ml-5">
						<div class="d-flex justify-content-between">
							<div>${replyReply.content}</div>
							<div class="d-flex align-items-start">
								<div class="font-italic badge">작성자: ${replyReply.user.username} &nbsp;</div>
								<c:if test="${replyReply.user.id == principal.user.id}">
										<button onClick="index.replyReplyDelete(${board.id}, ${reply.id}, ${replyReply.id})" class="badge">삭제</button>
								</c:if>
							</div>
						</div>
					</li>
				</c:forEach>
			</c:forEach>
		</ul>
	</div>

</div>

<script src="/js/board.js"></script>

<%@ include file="../layout/footer.jsp"%>